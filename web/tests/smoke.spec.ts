import { test, expect } from "@playwright/test";
test("upload renders results", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Upload car images")).toBeVisible();
});
