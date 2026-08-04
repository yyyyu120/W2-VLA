import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the W²-VLA project page", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /W²-VLA/);
  assert.match(html, /World-to-Wrist/);
  assert.match(html, /98\.5/);
  assert.match(html, /RoboTwin/);
  assert.match(html, /world-to-wrist-mute-16x9\.mp4/);
  assert.match(html, /poster="\/assets\/figures\/teaser\.png"/);
  assert.match(html, /task1_table_cleaning\/normal_demo\.mp4/);
  assert.match(html, /Three tasks across four visual conditions/);
  assert.match(html, /Vision-language-action \(VLA\) models often treat main-view/);
  assert.doesNotMatch(html, /See how it works/);
  assert.doesNotMatch(html, /Conventional multi-view VLAs treat camera views in parallel/);
  assert.doesNotMatch(html, /The world branch captures global intent/);
  assert.doesNotMatch(html, /The four source tables are presented/);
  assert.doesNotMatch(html, /Toggle the condition and metric/);
  assert.doesNotMatch(html, /All figures used in the main text/);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|Your site is taking shape/);
});
