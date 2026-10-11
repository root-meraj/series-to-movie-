"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ROOT = path.resolve(__dirname, "..");
const DEFAULT_LIMIT = 2 * 1024 ** 3;

// This intentionally models only the DOM used by the upload/connection flow.
// All production event listeners and rendering functions still run unchanged.
class Element {
  constructor(document, tagName, attributes, owner = null) {
    this.document = document;
    this.tagName = tagName.toUpperCase();
    this.attributes = attributes;
    this.owner = owner;
    this.parentElement = owner;
    this.listeners = new Map();
    this.value = attributes.value || "";
    this.textContent = "";
    this.hidden = "hidden" in attributes;
    this.disabled = "disabled" in attributes;
    this.isConnected = true;
    this.open = false;
    this.style = {};
    this.files = [];
    this.dataset = Object.fromEntries(Object.entries(attributes)
      .filter(([name]) => name.startsWith("data-"))
      .map(([name, value]) => [name.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase()), value]));
    const classes = new Set((attributes.class || "").split(/\s+/).filter(Boolean));
    this.classList = {
      contains: name => classes.has(name),
      add: name => classes.add(name),
      remove: name => classes.delete(name),
      toggle(name, force = !classes.has(name)) {
        if (force) classes.add(name); else classes.delete(name);
        return force;
      },
    };
  }
  set innerHTML(value) {
    this.markup = value;
    this.textContent = String(value).replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();
    this.document.parse(value, this);
  }
  get innerHTML() { return this.markup || ""; }
  addEventListener(name, callback) {
    const listeners = this.listeners.get(name) || [];
    listeners.push(callback);
    this.listeners.set(name, listeners);
  }
  async emit(name, fields = {}) {
    const event = { target: this, preventDefault() {}, ...fields };
    for (const listener of this.listeners.get(name) || []) await listener(event);
  }
  querySelector(selector) { return this.document.querySelector(selector, this); }
  querySelectorAll(selector) { return this.document.querySelectorAll(selector, this); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  removeAttribute(name) { delete this.attributes[name]; }
  remove() { this.isConnected = false; }
  showModal() { this.open = true; }
  close() { this.open = false; this.emit("close"); }
  focus() { this.document.activeElement = this; }
  pause() {}
  load() {}
}

class Document {
  constructor(markup) {
    this.elements = [];
    this.listeners = new Map();
    this.hidden = false;
    this.parse(markup);
    this.body = this.querySelector("body");
  }
  parse(markup, owner = null) {
    if (owner) for (const element of this.elements) {
      if (this.within(element, owner)) element.isConnected = false;
    }
    for (const match of markup.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)) {
      const attributes = {};
      for (const attribute of match[2].matchAll(/([^\s=]+)(?:="([^"]*)"|='([^']*)'|=([^\s]+))?/g)) {
        attributes[attribute[1]] = attribute[2] ?? attribute[3] ?? attribute[4] ?? "";
      }
      const element = new Element(this, match[1], attributes, owner);
      // Store static text so button/caption assertions test the actual rendered copy.
      const following = markup.slice(match.index + match[0].length);
      const close = following.indexOf(`</${match[1]}>`);
      if (close >= 0) element.textContent = following.slice(0, close)
        .replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();
      this.elements.push(element);
    }
  }
  within(element, owner) {
    for (let parent = element.owner; parent; parent = parent.owner) if (parent === owner) return true;
    return false;
  }
  matches(element, selector) {
    if (selector.startsWith("#")) return element.attributes.id === selector.slice(1);
    if (selector.startsWith(".")) return element.classList.contains(selector.slice(1));
    const attribute = /^\[([^=\]]+)(?:="([^"]*)")?\]$/.exec(selector);
    if (attribute) return attribute[1] in element.attributes
      && (attribute[2] === undefined || element.attributes[attribute[1]] === attribute[2]);
    return element.tagName.toLowerCase() === selector;
  }
  querySelectorAll(selector, owner = null) {
    return this.elements.filter(element => element.isConnected && (!owner || this.within(element, owner))
      && selector.split(",").some(part => this.matches(element, part.trim())));
  }
  querySelector(selector, owner = null) { return this.querySelectorAll(selector, owner)[0] || null; }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  createElement(tagName) { return new Element(this, tagName, {}); }
}

function frontend() {
  const document = new Document(fs.readFileSync(path.join(ROOT, "web", "index.html"), "utf8"));
  const fetches = [];
  const uploads = [];
  const responses = [];
  const location = { protocol: "http:", hash: "#new", replace(value) { this.hash = value; } };
  const context = {
    document, location, URL, AbortController,
    window: { STS_CONFIG: {}, addEventListener() {}, scrollTo() {} },
    localStorage: { getItem() { return ""; }, setItem() {} },
    setTimeout() { return 1; }, clearTimeout() {},
    fetch: async (url, options) => {
      fetches.push({ url, options });
      assert.ok(responses.length, "Unexpected request: the harness never uses a real network");
      const payload = responses.shift();
      return { ok: true, status: 200, json: async () => payload };
    },
    FormData: class { constructor() { this.entries = []; } append(...entry) { this.entries.push(entry); } },
    XMLHttpRequest: class {
      constructor() { uploads.push(this); throw new Error("Unexpected upload in a validation-only flow"); }
    },
  };
  const source = fs.readFileSync(path.join(ROOT, "web", "app.js"), "utf8");
  const end = source.lastIndexOf("})();");
  assert.ok(end >= 0, "The frontend must retain its encapsulating closure");
  // Expose only state for assertions; execute every actual function and startup statement.
  const instrumented = source.slice(0, end) + "globalThis.frontendState = app;\n" + source.slice(end);
  vm.runInNewContext(instrumented, context, { filename: "web/app.js" });
  const element = selector => {
    const found = document.querySelector(selector);
    assert.ok(found, `Missing UI element ${selector}`);
    return found;
  };
  return {
    app: context.frontendState, element, uploads, fetches, location,
    async choose(name, size) {
      const file = { name, size };
      element("#archive").files = [file];
      await element("#archive").emit("change");
      return file;
    },
    async submit() { await element("#create-form").emit("submit"); },
    async connect(limit = DEFAULT_LIMIT) {
      element("#api-url").value = "http://workspace.invalid";
      element("#access-key").value = "synthetic-workspace-key";
      responses.push({ status: "ok", max_upload_bytes: limit, default_target_minutes: 120 }, []);
      await element("#connect-form").emit("submit");
      assert.equal(element("#connect-error").hidden, true, element("#connect-error").textContent);
      assert.equal(context.frontendState.connected, true);
    },
  };
}

const cases = {
  async "small-disconnected"() {
    const ui = frontend();
    const file = await ui.choose("archive.zip", 1024 ** 2);
    assert.equal(ui.app.file, file);
    assert.equal(ui.element("#file-name").textContent, "archive.zip");
    assert.equal(ui.element("#create-error").hidden, true);
    assert.match(ui.element("#create-submit").textContent, /connect/i);
    assert.match(ui.element("#file-description").textContent, /connect/i);
    await ui.submit();
    assert.equal(ui.element("#connect-dialog").open, true);
    assert.equal(ui.app.file, file);
    assert.equal(ui.uploads.length, 0);
    assert.equal(ui.fetches.length, 0);
  },
  async "oversize-replacement"() { await rejectedReplacement("oversize.zip", DEFAULT_LIMIT + 1, /2 GB/); },
  async "empty-replacement"() { await rejectedReplacement("empty.zip", 0, /empty/i); },
  async "nonzip-replacement"() { await rejectedReplacement("episode.mp4", 1024, /ZIP/i); },
  async "inclusive-size-boundary"() {
    const ui = frontend();
    const file = await ui.choose("at-limit.ZIP", DEFAULT_LIMIT);
    assert.equal(ui.app.file, file);
    assert.equal(ui.element("#create-error").hidden, true);
    await ui.choose("too-large.zip", DEFAULT_LIMIT + 1);
    assert.equal(ui.app.file, null);
    assert.match(ui.element("#create-error").textContent, /2 GB/);
  },
  async "connected-limit"() {
    const ui = frontend();
    ui.app.connected = true;
    ui.app.health = { max_upload_bytes: 512 * 1024 };
    await ui.choose("too-large.zip", 1024 ** 2);
    assert.equal(ui.app.file, null);
    const error = ui.element("#create-error").textContent;
    assert.equal(ui.element("#create-error").hidden, false);
    await ui.submit();
    assert.equal(ui.element("#create-error").textContent, error);
    assert.equal(ui.uploads.length, 0);
  },
  async "connect-preserves-form"() {
    const ui = frontend();
    const file = await ui.choose("archive.zip", 1024 ** 2);
    ui.element("#project-name").value = "My chosen title";
    ui.element("#target-minutes").value = "90";
    await ui.submit();
    await ui.connect();
    assert.equal(ui.app.file, file);
    assert.equal(ui.element("#project-name").value, "My chosen title");
    assert.equal(ui.element("#target-minutes").value, "90");
    assert.equal(ui.location.hash, "#new");
    assert.match(ui.element("#create-submit").textContent, /create/i);
    assert.doesNotMatch(ui.element("#file-description").textContent, /connect/i);
    assert.equal(ui.element("#create-error").hidden, true);
    assert.equal(ui.uploads.length, 0, "Connecting does not silently submit the archive");
    assert.deepEqual(ui.fetches.map(entry => new URL(entry.url).pathname), ["/api/health", "/api/jobs"]);
  },
  async "connect-refreshes-limit"() {
    const ui = frontend();
    const file = await ui.choose("archive.zip", 1024 ** 2);
    await ui.connect(512 * 1024);
    assert.equal(ui.app.selectedFile, file, "Keep the user's candidate for limit revalidation");
    assert.equal(ui.app.file, null);
    assert.equal(ui.element("#file-name").textContent, "archive.zip");
    assert.match(ui.element("#create-error").textContent, /limit|under|exceed/i);
    assert.equal(ui.element("#create-error").hidden, false);
    assert.doesNotMatch(ui.element("#file-description").textContent, /ready to upload/i);
    assert.doesNotMatch(ui.element(".upload-meta").textContent, /2 GB/);
    await ui.submit();
    assert.equal(ui.uploads.length, 0);
  },
  async "reconnect-clears-size-error"() {
    const ui = frontend();
    const file = await ui.choose("archive.zip", 1024 ** 2);
    await ui.connect(512 * 1024);
    assert.equal(ui.element("#create-error").hidden, false);
    await ui.connect(DEFAULT_LIMIT);
    assert.equal(ui.app.file, file);
    assert.equal(ui.element("#create-error").hidden, true);
    assert.match(ui.element("#file-description").textContent, /ready/i);
    assert.match(ui.element(".upload-meta").textContent, /2 GB/);
  },
};

async function rejectedReplacement(name, size, message) {
  const ui = frontend();
  await ui.choose("archive.zip", 1024 ** 2);
  await ui.choose(name, size);
  assert.equal(ui.app.file, null);
  assert.notEqual(ui.element("#file-name").textContent, "archive.zip");
  assert.doesNotMatch(ui.element("#file-description").textContent, /ready to upload/i);
  assert.match(ui.element("#create-error").textContent, message);
  const error = ui.element("#create-error").textContent;
  await ui.submit();
  assert.equal(ui.element("#create-error").textContent, error);
  assert.equal(ui.uploads.length, 0);
}

const requested = process.argv[2];
assert.ok(!requested || Object.hasOwn(cases, requested), `Unknown frontend case: ${requested}`);
for (const [name, run] of Object.entries(cases)) if (!requested || requested === name) test(name, run);
