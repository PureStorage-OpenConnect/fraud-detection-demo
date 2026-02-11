# Security Review: pravee42/aiuc.spearhead.so

**Repository:** https://github.com/pravee42/aiuc.spearhead.so
**Date:** 2026-02-11
**Reviewer:** Automated security review via Claude
**Application Type:** Next.js 14 data analytics dashboard (AWS S3-backed, deployed on AWS Amplify)

---

## Executive Summary

This repository contains a Next.js application that displays AI use case data in a tabular interface, pulling data from AWS S3. The review identified **4 critical/high severity**, **5 medium severity**, and **several low severity** security issues. The most serious finding is that **authentication is explicitly disabled** in the middleware while the application footer labels the data as "Confidential - Internal Use Only."

---

## Critical / High Severity

### 1. CRITICAL — Authentication Completely Disabled

**File:** `middleware.ts`

```typescript
export function middleware(request: NextRequest) {
  // Authentication disabled
  return NextResponse.next();
}
```

The middleware intercepts all non-static routes but performs zero authentication or authorization. Every API endpoint and page is publicly accessible to anyone on the internet.

**Impact:** Complete unauthorized access to all data and API endpoints. The footer states "Confidential - Internal Use Only," creating a direct contradiction — confidential data is served without any access control.

**Recommendation:** Implement authentication. Options include:
- Next.js middleware-based token/session validation
- NextAuth.js for OAuth/SSO integration
- AWS Cognito (already within the AWS ecosystem)
- At minimum, a shared secret / API key for the API routes

---

### 2. HIGH — No API Authentication on S3 Data Endpoints

**Files:** `app/api/use-cases/route.ts`, `app/api/industry-datas/route.ts`

Both API routes serve data from S3 with no authentication check whatsoever. Any HTTP client can call:
- `GET /api/use-cases?page=1&page_size=99999`
- `GET /api/industry-datas?page=1&page_size=99999`

...and retrieve the entire dataset.

**Impact:** Full data exfiltration by any unauthenticated party.

**Recommendation:** Add authentication middleware or per-route authorization checks before serving data.

---

### 3. HIGH — AWS Credential Handling Pattern

**Files:** `app/api/use-cases/route.ts`, `app/api/industry-datas/route.ts`

```typescript
const s3Client = new S3Client({
    region: process.env.AWS_REGION || 'us-east-1',
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
    },
});
```

Issues:
- Explicit credential injection bypasses the AWS SDK default credential chain (IAM roles, instance profiles, etc.)
- Empty-string fallbacks (`|| ''`) cause silent authentication failures instead of clear errors
- Forces long-lived access keys instead of allowing temporary credentials via IAM roles

**Recommendation:**
- On AWS Amplify/EC2/ECS: Remove explicit credentials entirely and rely on IAM roles
- If access keys are necessary, remove the `|| ''` fallbacks so missing credentials fail loudly
- Use `@aws-sdk/credential-providers` for more secure credential resolution

---

### 4. HIGH — Diagnostic Script Exposes Infrastructure Details

**File:** `diagnose-s3.js`

This file is committed to the **public** repository and reveals:
- The default S3 bucket name (`'aiuc'`)
- The AWS region (`'ap-southeast-2'`)
- The data structure of all S3 objects (JSON parsing logic)
- Credential loading via `dotenv`

```javascript
const s3Client = new S3Client({
    region: process.env.AWS_REGION || 'ap-southeast-2',
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    },
});
```

Note the region inconsistency: `diagnose-s3.js` defaults to `ap-southeast-2` while the API routes default to `us-east-1`. This suggests a configuration drift that could itself be exploitable.

**Impact:** Attackers gain knowledge of the S3 bucket name, region, and data schema, reducing the effort needed for targeted attacks.

**Recommendation:**
- Remove `diagnose-s3.js` from the repository
- Add it to `.gitignore`
- Audit git history to ensure no `.env` file was ever committed (use `git log --all --full-history -- .env`)

---

## Medium Severity

### 5. MEDIUM — No Input Validation on Pagination Parameters

**Files:** `app/api/use-cases/route.ts`, `app/api/industry-datas/route.ts`

```typescript
const page = parseInt(searchParams.get('page') || '1');
const pageSize = parseInt(searchParams.get('page_size') || '500');
```

No validation is performed on these values:
- Negative numbers, zero, NaN, or extremely large values are silently accepted
- `page_size=999999999` forces the server to allocate and serialize a massive JSON response
- `parseInt('abc')` returns `NaN`, causing `slice(NaN, NaN)` to return the entire array

**Impact:** Memory exhaustion (DoS), unexpected data exposure from NaN edge cases.

**Recommendation:**
```typescript
const page = Math.max(1, parseInt(searchParams.get('page') || '1') || 1);
const pageSize = Math.min(1000, Math.max(1, parseInt(searchParams.get('page_size') || '20') || 20));
```

---

### 6. MEDIUM — No Rate Limiting

Every API call triggers a fresh `GetObjectCommand` to S3 (due to `force-dynamic`). There is no rate limiting at the application level.

**Impact:**
- Financial: Each request incurs S3 GET costs. An attacker can generate thousands of requests
- Availability: Can overwhelm the server with S3 calls, causing latency spikes or timeouts
- Bandwidth: S3 egress costs from repeated full-dataset fetches

**Recommendation:**
- Add rate limiting middleware (e.g., `next-rate-limit` or Vercel/Amplify edge rate limits)
- Cache S3 responses in memory or via a CDN with a reasonable TTL
- Remove `force-dynamic` if real-time data freshness isn't required

---

### 7. MEDIUM — No S3 Response Validation

```typescript
const s3Data = JSON.parse(bodyContents);
const useCaseDataRaw: UseCaseDataRaw[] = s3Data.data;
```

The code trusts S3 data completely:
- No schema validation on the parsed JSON
- No type runtime checking (TypeScript interfaces only exist at compile time)
- If the S3 object is corrupted, malformed, or tampered with, the server may crash or serve garbage data
- The `streamToString` function accumulates the entire S3 object in memory with no size limit

**Impact:** Denial of service via large/corrupted S3 objects; potential data integrity issues.

**Recommendation:**
- Validate the parsed JSON against a schema (e.g., Zod, ajv)
- Set a maximum expected size for S3 responses
- Add a timeout for S3 operations

---

### 8. MEDIUM — Missing Security Headers

Neither `middleware.ts` nor `next.config.js` configures security headers:

| Missing Header | Risk |
|---|---|
| `Content-Security-Policy` | XSS protection |
| `X-Content-Type-Options: nosniff` | MIME type sniffing attacks |
| `X-Frame-Options: DENY` | Clickjacking |
| `Strict-Transport-Security` | Downgrade attacks |
| `Referrer-Policy` | Information leakage |
| `Permissions-Policy` | Feature abuse |

**Recommendation:** Add security headers in `next.config.js`:
```javascript
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  async headers() {
    return [{
      source: '/(.*)',
      headers: [
        { key: 'X-Content-Type-Options', value: 'nosniff' },
        { key: 'X-Frame-Options', value: 'DENY' },
        { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
        { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
      ],
    }];
  },
};
```

---

### 9. MEDIUM — No CORS Policy

No explicit CORS configuration exists. Next.js same-origin defaults may apply for browser requests, but:
- API routes can be called by any server-side client
- Without explicit CORS headers, the policy depends on the deployment platform
- If deployed behind a permissive reverse proxy, cross-origin browser requests may succeed

**Recommendation:** Explicitly configure CORS in the API routes or middleware to restrict access to expected origins.

---

## Low Severity

### 10. LOW — Error Message Information Leakage

```typescript
console.error('API Error:', error);
```

Full error objects (potentially containing S3 bucket ARNs, credential errors, or stack traces) are logged server-side. While the client receives a generic `"Internal server error"`, log aggregation services may expose this.

**Recommendation:** Log structured error data with sensitive fields redacted.

---

### 11. LOW — Untyped Stream Parameter

```typescript
async function streamToString(stream: any): Promise<string> {
```

The `any` type bypasses TypeScript's safety guarantees. This function also:
- Has no timeout
- Has no size limit on accumulated chunks
- Could hang indefinitely on a stalled stream

**Recommendation:** Type the parameter as `Readable` from Node.js streams, add a timeout, and set a max buffer size.

---

### 12. LOW — S3 Bucket Name Hardcoded in Source

```typescript
Bucket: process.env.BUCKET_NAME || 'aiuc',
```

The default bucket name `aiuc` is visible in the public source code, giving attackers a target for S3 enumeration or misconfiguration attacks.

**Recommendation:** Remove the hardcoded fallback; require `BUCKET_NAME` as a mandatory environment variable.

---

### 13. LOW — Next.js 14.0.0 May Have Known CVEs

The project uses `next@14.0.0` (the initial 14.x release). Several security patches have been released since then, including fixes for:
- Server-side request forgery via Server Actions
- Cache poisoning issues
- Middleware bypass vulnerabilities

**Recommendation:** Update to the latest Next.js 14.x patch release. Run `npm audit` regularly.

---

### 14. INFO — Region Configuration Inconsistency

`diagnose-s3.js` defaults to `ap-southeast-2` while the API routes default to `us-east-1`. This suggests either:
- The application was migrated between regions
- Different environments use different regions
- Configuration drift that could cause silent failures

---

## Summary Table

| # | Severity | Finding | File(s) |
|---|----------|---------|---------|
| 1 | CRITICAL | Authentication disabled | `middleware.ts` |
| 2 | HIGH | No API authentication | `app/api/*/route.ts` |
| 3 | HIGH | Insecure AWS credential handling | `app/api/*/route.ts` |
| 4 | HIGH | Diagnostic script in public repo | `diagnose-s3.js` |
| 5 | MEDIUM | No pagination input validation | `app/api/*/route.ts` |
| 6 | MEDIUM | No rate limiting | Application-wide |
| 7 | MEDIUM | No S3 response validation | `app/api/*/route.ts` |
| 8 | MEDIUM | Missing security headers | `middleware.ts`, `next.config.js` |
| 9 | MEDIUM | No CORS policy | Application-wide |
| 10 | LOW | Error info leakage in logs | `app/api/*/route.ts` |
| 11 | LOW | Untyped stream handling | `app/api/*/route.ts` |
| 12 | LOW | Hardcoded S3 bucket name | `app/api/*/route.ts`, `diagnose-s3.js` |
| 13 | LOW | Outdated Next.js version | `package.json` |
| 14 | INFO | Region configuration inconsistency | `diagnose-s3.js` vs `app/api/*/route.ts` |

---

## Priority Remediation Order

1. **Immediately** enable authentication in `middleware.ts`
2. **Immediately** remove `diagnose-s3.js` from the repository and audit git history
3. **Soon** add input validation to pagination parameters
4. **Soon** add security headers and CORS configuration
5. **Soon** refactor AWS credential handling to use IAM roles
6. **Planned** add rate limiting, S3 response caching, and schema validation
7. **Planned** update Next.js to latest patch version
