import type { NextConfig } from "next";

/**
 * FuturesRL dashboard build configuration.
 *
 * - `output: "standalone"` keeps the production Docker image small (only the
 *   server bundle + pruned node_modules are copied into the runner stage).
 * - No remote hosts are trusted: the dashboard talks to the FastAPI backend
 *   through `NEXT_PUBLIC_API_BASE_URL` over REST/WebSocket at runtime.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Docker images set NEXT_OUTPUT=standalone; local `next start` works without it.
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
  eslint: {
    // `npm run lint` is the dedicated lint gate; the production build still
    // fails on TypeScript errors via `next build`'s native type check.
    ignoreDuringBuilds: true,
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
