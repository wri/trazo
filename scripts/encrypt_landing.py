#!/usr/bin/env python3
"""Encrypt the Trazo landing page with AES-GCM, wrapped in a password prompt.

Reads the plaintext source (``_landing-src/index.html`` — the leading
underscore keeps Jekyll from publishing it) and writes a self-contained
password-prompt page to ``landing/index.html``. The output has no network
dependencies and decrypts client-side via the Web Crypto API, so it can be
served from any static host (here: GitHub Pages at /trazo/landing/).

Why client-side encryption rather than server-side auth: the ``wri/trazo``
Pages site is public, so anything deployed there is world-readable. Encrypting
the HTML lets us share the page via URL + password without standing up a login.
Tradeoff: anyone with the ciphertext + password can read it, and anyone with the
ciphertext alone can run an offline crack attack — so use a strong password and
rotate it when sharing ends. (A short/dictionary password offers little real
protection.)

Usage:
    TRAZO_SITE_PASSWORD='...' /usr/bin/python3 scripts/encrypt_landing.py

Re-run after editing the source HTML or to rotate the password, then commit the
regenerated landing/index.html.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PBKDF2_ITERATIONS = 300_000

SOURCE = Path("_landing-src/index.html")
OUTPUT = Path("landing/index.html")

PAGE_TITLE = "Trazo"
CARD_TITLE = "Trazo"
CARD_SUBTITLE = "Private preview"


_WRAPPER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__PAGE_TITLE__</title>
<link rel="stylesheet" href="https://use.typekit.net/fxq7ozw.css">
<style>
  :root {
    --gold: #F0AB00;
    --bg-deep: #12100b;
    --bg-panel: #1a1712;
    --text: #EDE7D9;
    --text-muted: #A89F8C;
    --border: rgba(240,171,0,0.30);
    --font-header: 'acumin-pro', 'Arial Narrow', Arial, sans-serif;
    --font-body: 'adobe-caslon-pro', Georgia, serif;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: var(--font-body);
    background: var(--bg-deep);
    color: var(--text);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
  }
  .card {
    max-width: 400px;
    width: 100%;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    padding: 2.6rem 2rem 2rem;
  }
  .dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--gold);
    margin-right: 10px;
    vertical-align: middle;
  }
  h1 {
    font-family: var(--font-header);
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 0.4rem 0;
  }
  .sub {
    font-family: var(--font-header);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 1.6rem;
  }
  label {
    display: block;
    font-family: var(--font-header);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
  }
  input[type=password] {
    width: 100%;
    padding: 0.7rem 0.8rem;
    font-size: 0.95rem;
    font-family: var(--font-body);
    border: 1px solid var(--border);
    background: #0e0c08;
    color: var(--text);
    margin-bottom: 1rem;
  }
  input[type=password]:focus {
    outline: none;
    border-color: var(--gold);
    box-shadow: 0 0 0 2px rgba(240,171,0,0.18);
  }
  button {
    width: 100%;
    padding: 0.75rem 1rem;
    background: var(--gold);
    color: #1a1712;
    border: none;
    font-family: var(--font-header);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover:not(:disabled) { background: #ffd04d; }
  button:disabled { opacity: 0.6; cursor: wait; }
  .err {
    margin-top: 0.85rem;
    font-size: 0.9rem;
    color: #ff8a6a;
    min-height: 1.2em;
  }
  .footer {
    margin-top: 2rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.5;
  }
</style>
</head>
<body>
<div class="card">
  <h1><span class="dot"></span>__CARD_TITLE__</h1>
  <div class="sub">__CARD_SUBTITLE__</div>
  <form id="pw-form">
    <label for="pw">Access password</label>
    <input type="password" id="pw" autocomplete="off" autofocus>
    <button id="submit-btn" type="submit">Enter</button>
    <div class="err" id="err"></div>
  </form>
  <div class="footer">
    Encrypted client-side. The password decrypts the page in your browser;
    nothing is sent to a server.
  </div>
</div>
<script>
  const META = __META__;

  const fromB64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

  async function decrypt(password) {
    const enc = new TextEncoder();
    const baseKey = await crypto.subtle.importKey(
      "raw", enc.encode(password),
      { name: "PBKDF2" }, false, ["deriveKey"]
    );
    const aesKey = await crypto.subtle.deriveKey(
      {
        name: "PBKDF2",
        salt: fromB64(META.salt),
        iterations: META.iterations,
        hash: "SHA-256",
      },
      baseKey,
      { name: "AES-GCM", length: 256 },
      false, ["decrypt"]
    );
    const plaintext = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: fromB64(META.nonce) },
      aesKey,
      fromB64(META.ciphertext)
    );
    return new TextDecoder().decode(plaintext);
  }

  document.getElementById("pw-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const pw = document.getElementById("pw").value;
    const err = document.getElementById("err");
    const btn = document.getElementById("submit-btn");
    err.textContent = "";
    btn.disabled = true;
    btn.textContent = "Decrypting...";
    try {
      const html = await decrypt(pw);
      // The decrypted page brings its own <html>/<head>/<body>;
      // document.open()/write()/close() swaps the whole document in place.
      // The prompt re-appears on refresh, since the password isn't stored.
      document.open();
      document.write(html);
      document.close();
    } catch (e2) {
      err.textContent = "Wrong password.";
      btn.disabled = false;
      btn.textContent = "Enter";
      document.getElementById("pw").select();
    }
  });
</script>
</body>
</html>
"""


def main() -> None:
    password = os.environ.get("TRAZO_SITE_PASSWORD")
    if not password:
        sys.exit(
            "TRAZO_SITE_PASSWORD not set. Pass it via env var:\n"
            "  TRAZO_SITE_PASSWORD='...' /usr/bin/python3 scripts/encrypt_landing.py"
        )
    if len(password) < 8:
        print(
            f"WARNING: password is only {len(password)} chars — weak against an "
            "offline crack of the public ciphertext. Rotate to something stronger "
            "before sharing widely.",
            file=sys.stderr,
        )

    if not SOURCE.exists():
        sys.exit(f"Source HTML not found: {SOURCE}")

    plaintext = SOURCE.read_bytes()

    salt = secrets.token_bytes(16)
    from hashlib import pbkdf2_hmac

    key = pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS, 32)
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)

    def b64(b: bytes) -> str:
        return base64.b64encode(b).decode()

    meta = {
        "salt": b64(salt),
        "nonce": b64(nonce),
        "ciphertext": b64(ciphertext),
        "iterations": PBKDF2_ITERATIONS,
    }

    html = (
        _WRAPPER
        .replace("__PAGE_TITLE__", PAGE_TITLE)
        .replace("__CARD_TITLE__", CARD_TITLE)
        .replace("__CARD_SUBTITLE__", CARD_SUBTITLE)
        .replace("__META__", json.dumps(meta))
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")

    src_kb = len(plaintext) / 1024
    out_kb = OUTPUT.stat().st_size / 1024
    print(f"Encrypted {SOURCE} ({src_kb:,.0f} KB) -> {OUTPUT} ({out_kb:,.0f} KB)")
    print(f"PBKDF2 iterations: {PBKDF2_ITERATIONS:,}")


if __name__ == "__main__":
    main()
