import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://localhost:3000" },
  webServer: {
    command: "npm run dev",
    port: 3000,
    reuseExistingServer: true,
    timeout: 180 * 1000,
    // The demo gates its fetch path on NEXT_PUBLIC_API_URL; mock routes intercept
    // the request, so any host works — this keeps tests green without shell env.
    env: { NEXT_PUBLIC_API_URL: "http://localhost:8000" },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
