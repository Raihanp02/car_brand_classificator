import { test, expect } from "@playwright/test";
import path from "path";

const samplesDir = path.join(__dirname, "../public/samples");
const sample = (n: number) => path.join(samplesDir, `sample-car-${n}.jpg`);

function top5(prefix: string) {
  return Array.from({ length: 5 }, (_, i) => ({
    label: `${prefix}-class-${i + 1}`,
    score: 0.9 - i * 0.1,
  }));
}

test("single upload renders 1 result card with top-5 details", async ({ page }) => {
  await page.route("**/v1/predict/batch", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        results: [
          {
            filename: "sample-car-1.jpg",
            label: "audi-a4",
            confidence: 0.92,
            top5: top5("audi"),
          },
        ],
      }),
    });
  });
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(sample(1));
  const cards = page.getByTestId("result-card");
  await expect(cards).toHaveCount(1);
  await expect(cards.first().getByText("audi-a4")).toBeVisible();
  const details = cards.first().locator("details");
  await expect(details).toBeVisible();
  await details.locator("summary").click();
  await expect(details.locator("ol li")).toHaveCount(5);
});

test("try sample cars renders 3 result cards with previews", async ({ page }) => {
  await page.route("**/v1/predict/batch", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        // Duplicate filename included deliberately: keys/previews are index-based.
        results: [
          { filename: "sample-car-1.jpg", label: "audi-a4", confidence: 0.9, top5: top5("a") },
          { filename: "sample-car-1.jpg", label: "bmw-3", confidence: 0.8, top5: top5("b") },
          { filename: "sample-car-3.jpg", label: "vw-golf", confidence: 0.7, top5: top5("c") },
        ],
      }),
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Try sample cars" }).click();
  const cards = page.getByTestId("result-card");
  await expect(cards).toHaveCount(3);
  // Index-based preview mapping: every card (incl. the duplicate name) has an image.
  await expect(cards.locator("img")).toHaveCount(3);
  await expect(cards.nth(1).getByText("bmw-3")).toBeVisible();
  await cards.first().locator("details summary").click();
  await expect(cards.first().locator("details ol li")).toHaveCount(5);
});

test("contract variants render gracefully (top_5, clamped confidence)", async ({ page }) => {
  await page.route("**/v1/predict/batch", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        results: [{ filename: "odd.jpg", label: "fiat-500", confidence: 1.7, top_5: top5("f") }],
      }),
    });
  });
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(sample(2));
  const cards = page.getByTestId("result-card");
  await expect(cards).toHaveCount(1);
  // Confidence clamped to 100%, top_5 coerced — no NaN% and no crash.
  await expect(cards.first().getByText("fiat-500")).toBeVisible();
  await expect(cards.first().getByText("100%")).toBeVisible();
  await expect(cards.first().locator("details ol li")).toHaveCount(5);
});
