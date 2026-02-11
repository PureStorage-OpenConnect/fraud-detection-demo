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

---
---

# Security Review: pravee42/aiuc.spearhead.so — `react-version` Branch

**Repository:** https://github.com/pravee42/aiuc.spearhead.so
**Branch:** `react-version`
**Date:** 2026-02-11
**Reviewer:** Automated security review via Claude
**Application Type:** React 19 + Vite 7 SPA (client-side only, deployed on AWS Amplify)

---

## Executive Summary

The `react-version` branch is a **complete architectural rewrite** from the `main` branch. It replaces the Next.js server-side application with a **pure client-side React SPA** (Vite + React 19). This eliminates the server-side API proxy layer entirely, meaning the browser fetches data directly from S3 via a publicly accessible URL.

This architecture is **significantly less secure** than the `main` branch. The server-side layer that previously kept AWS credentials private is gone — instead, the S3 bucket must be publicly readable, and its URL is embedded directly in the JavaScript bundle shipped to every user's browser.

**11 findings identified:** 3 critical/high, 5 medium, 3 low.

---

## Architecture Comparison: main vs react-version

| Aspect | main (Next.js) | react-version (Vite SPA) |
|--------|---------------|--------------------------|
| Architecture | Server-side API proxy | Pure client-side SPA |
| S3 Access | Server fetches from S3 (creds server-only) | Browser fetches from S3 directly |
| Authentication | Disabled (but middleware exists) | **Impossible** (no server layer) |
| S3 Credentials | Server env vars (not exposed to client) | Bucket must be **public** |
| Data Exposure | Through API endpoints | Direct S3 access from browser |
| Security Headers | Could be added in middleware | Must rely entirely on hosting platform |
| XSS Surface | React auto-escaping (server-rendered) | React auto-escaping + **unvalidated URLs** |

---

## Critical / High Severity

### 1. CRITICAL — S3 Bucket URL Exposed in Client-Side JavaScript Bundle

**File:** `src/hooks/useS3Data.ts`

```typescript
const S3_BASE_URL = import.meta.env.VITE_S3_BASE_URL;
// ...
const response = await fetch(`${S3_BASE_URL}/use_cases.json`);
```

Vite environment variables prefixed with `VITE_` are **compiled directly into the production JavaScript bundle**. This means:

- The full S3 bucket URL is visible to anyone who opens browser DevTools or views the page source
- The exact JSON file names (`use_cases.json`, `industry_use_cases.json`) are exposed
- Anyone can download the raw data directly from S3 without even visiting the application
- The S3 bucket **must** be configured for public read access (or have permissive CORS) for this architecture to work

**Impact:** Complete exposure of the data source. Anyone can scrape all data directly from S3, bypassing the application entirely. There is no way to add authentication, rate limiting, or access logging at the application level.

**Recommendation:** This architecture fundamentally cannot protect data. If the data is confidential, you must:
- Re-introduce a server-side API proxy (revert to Next.js or add an API Gateway)
- Use AWS CloudFront with signed URLs or cookies for time-limited access
- At minimum, use AWS API Gateway with Cognito authentication in front of S3

---

### 2. CRITICAL — No Authentication Possible (Pure SPA Architecture)

**Entire application**

This is a pure client-side SPA with:
- No server-side component
- No authentication mechanism
- No login page
- No token management
- No middleware (not a Next.js app)
- No way to protect API calls

The footer still states **"Confidential - Internal Use Only"** but there is zero access control.

**Impact:** Anyone with the URL has full access to all data. Unlike the `main` branch (which at least has a middleware hook where authentication could be re-enabled), this architecture has no place to add authentication without a server.

**Recommendation:** If this data is truly confidential:
- Use AWS Amplify Authentication (Cognito) to gate access
- Or put the SPA behind a reverse proxy with authentication (e.g., CloudFront + Lambda@Edge)
- Or revert to the Next.js architecture which at least has middleware capability

---

### 3. HIGH — Potential XSS via Unvalidated URL Rendering

**File:** `src/components/IndustryDataTable.tsx`

```tsx
<a
  style={{
    textDecoration: "underline",
    color: PURE_ORANGE,
    display: "flex",
    alignItems: "center",
    gap: 4,
  }}
  href={getValue() as string}
  target="_blank"
  rel="noopener noreferrer"
>
  Visit Site <OpenInNewIcon sx={{ fontSize: 12 }} />
</a>
```

The "Industry References" column renders the cell value directly as an `href` attribute **without any URL validation**. If the S3 data (or a compromised S3 bucket) contains a malicious value like:

```
javascript:alert(document.cookie)
```

...clicking "Visit Site" would execute arbitrary JavaScript in the user's browser.

While `target="_blank"` with `rel="noopener noreferrer"` prevents reverse tabnapping, it does **not** prevent `javascript:` protocol XSS.

**Impact:** Stored XSS if the S3 data source is compromised or contains crafted URLs. This could lead to session hijacking, data theft, or phishing.

**Recommendation:**
```typescript
const sanitizeUrl = (url: string): string => {
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:'].includes(parsed.protocol)) return '#';
    return url;
  } catch {
    return '#';
  }
};

// Then use: href={sanitizeUrl(getValue() as string)}
```

---

## Medium Severity

### 4. MEDIUM — No Response Validation on Fetched S3 Data

**File:** `src/hooks/useS3Data.ts`

```typescript
const rawData = await response.json();
const mappedData = rawData.map((item: any) => ({...}));
```

The fetched JSON is parsed and mapped without any validation:
- No schema validation (the `any` type in `.map((item: any)` bypasses all safety)
- If the S3 data structure changes or is corrupted, the app may crash or display garbage
- A compromised S3 bucket could inject arbitrary data that gets rendered in the UI
- React auto-escapes JSX text content, but combined with finding #3, malicious data could still cause harm

**Impact:** Data integrity issues, potential for injection if combined with other vulnerabilities.

**Recommendation:** Validate fetched data with a runtime schema validator (Zod, valibot, etc.) before rendering.

---

### 5. MEDIUM — `--legacy-peer-deps` in Build Pipeline

**File:** `amplify.yml`

```yaml
preBuild:
  commands:
    - npm install --legacy-peer-deps
```

Using `--legacy-peer-deps` bypasses npm's peer dependency resolution, which:
- Can install incompatible dependency versions
- Masks security vulnerabilities that newer peer dependency requirements would catch
- May introduce subtle runtime bugs from version mismatches
- Indicates there are unresolved dependency conflicts in the project

**Impact:** Unknown vulnerable dependency versions may be silently installed.

**Recommendation:** Resolve the peer dependency conflicts properly rather than bypassing the check. Run `npm audit` to identify vulnerable packages.

---

### 6. MEDIUM — Missing Security Headers

**Files:** `vite.config.ts`, `index.html`

The Vite config has no security header configuration:

```typescript
export default defineConfig({
  plugins: [react()],
})
```

And `index.html` has no meta-tag security headers. Missing headers include:
- `Content-Security-Policy` (especially important since the app loads data from an external S3 URL)
- `X-Content-Type-Options`
- `X-Frame-Options`
- `Referrer-Policy`

For a pure SPA, these must be configured at the hosting level (Amplify custom headers) or via `<meta>` tags.

**Recommendation:** Add an Amplify `customHttp.yml` or configure headers in the Amplify console:
```yaml
customHeaders:
  - pattern: '**/*'
    headers:
      - key: X-Content-Type-Options
        value: nosniff
      - key: X-Frame-Options
        value: DENY
      - key: Content-Security-Policy
        value: "default-src 'self'; connect-src 'self' https://*.amazonaws.com; style-src 'self' 'unsafe-inline'; script-src 'self'"
```

---

### 7. MEDIUM — No Error Boundary

**File:** `src/main.tsx`

```typescript
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

No React Error Boundary wraps the application. If a rendering error occurs (e.g., from malformed S3 data), the entire app will crash with a white screen and potentially expose a stack trace in the browser console.

**Impact:** Poor error handling could reveal internal component structure; complete app crash from single data issue.

**Recommendation:** Add a top-level Error Boundary component.

---

### 8. MEDIUM — No CORS Restriction on S3 Fetch

**File:** `src/hooks/useS3Data.ts`

The app fetches from S3 with no credential restrictions:

```typescript
const response = await fetch(`${S3_BASE_URL}/use_cases.json`);
```

For this to work, the S3 bucket must have a permissive CORS policy (likely `Access-Control-Allow-Origin: *`). This means any website on the internet can also fetch this data via JavaScript.

**Impact:** Any malicious or third-party website can load and exfiltrate this data by making cross-origin requests to the same S3 bucket.

**Recommendation:** Restrict the S3 CORS policy to only allow the specific Amplify deployment domain.

---

## Low Severity

### 9. LOW — Development Tool Configs Committed

**Files:** `.cursor/worktrees.json`, `.vscode/settings.json`

While `.gitignore` lists these directories, they are still present in the repository (likely committed before being added to `.gitignore`). They reveal:
- Development tooling choices (Cursor AI, VS Code)
- Amplify directory structure patterns (via VS Code `files.exclude`)

**Recommendation:** Remove these files from tracking with `git rm --cached`.

---

### 10. LOW — Non-null Assertion on DOM Element

**File:** `src/main.tsx`

```typescript
document.getElementById('root')!
```

The `!` non-null assertion will cause a runtime crash if the `root` element doesn't exist, with no helpful error message.

**Recommendation:** Add a null check with a descriptive error.

---

### 11. INFO — Typo in Page Title

**File:** `index.html`

```html
<title>AIUC Spearehead</title>
```

"Spearehead" should be "Spearhead."

---

## Summary Table

| # | Severity | Finding | File(s) |
|---|----------|---------|---------|
| 1 | CRITICAL | S3 URL exposed in client JS bundle | `src/hooks/useS3Data.ts` |
| 2 | CRITICAL | No authentication possible (SPA) | Application-wide |
| 3 | HIGH | XSS via unvalidated URL in href | `src/components/IndustryDataTable.tsx` |
| 4 | MEDIUM | No response validation on S3 data | `src/hooks/useS3Data.ts` |
| 5 | MEDIUM | `--legacy-peer-deps` bypasses safety | `amplify.yml` |
| 6 | MEDIUM | Missing security headers | `vite.config.ts`, `index.html` |
| 7 | MEDIUM | No Error Boundary | `src/main.tsx` |
| 8 | MEDIUM | Permissive CORS required on S3 | `src/hooks/useS3Data.ts` |
| 9 | LOW | Dev tool configs committed | `.cursor/`, `.vscode/` |
| 10 | LOW | Non-null assertion on DOM element | `src/main.tsx` |
| 11 | INFO | Typo in page title | `index.html` |

---

## Key Takeaway: Architecture Regression

The `react-version` branch represents a **security regression** compared to `main`. The `main` branch (Next.js) at least has:
- A server-side layer where authentication _could_ be added
- Server-side S3 access that keeps credentials off the client
- Middleware where security headers and CORS can be configured

The `react-version` branch (pure SPA) **eliminates all of these capabilities**. If the data labeled "Confidential - Internal Use Only" is genuinely sensitive, this architecture cannot protect it.

## Priority Remediation Order

1. **Immediately** add URL sanitization to the Industry References `href` (XSS fix)
2. **Decide** whether this data requires protection — if yes, this architecture is unsuitable
3. **Soon** add runtime data validation on S3 responses
4. **Soon** configure security headers via Amplify custom headers
5. **Soon** restrict S3 CORS policy to the deployment domain only
6. **Planned** resolve dependency conflicts and remove `--legacy-peer-deps`
7. **Planned** clean up committed dev tool configs
