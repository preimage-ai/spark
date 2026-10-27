/**
 * Headless GLSL smoke test for the 3DGUT render mode.
 *
 * The gap this closes: nothing in either package's test suite ever links a GL
 * program. `tsc`, the 159 unit tests and `build:wasm` all pass with a shader
 * that will not compile, and a shader that fails to compile takes the whole
 * splat pass down rather than degrading. So "it builds" says nothing about
 * whether gutMode works, and the only honest check is to run it in a browser.
 *
 * This loads the viewer's own dev harness twice -- `?gut=0` then `?gut=1` --
 * and wraps the WebGL2 context to record every compileShader/linkProgram
 * outcome. three.js does log program errors itself, but it truncates the
 * info log and only fires on link failure, so the wrapper is what gives us
 * the driver's actual message plus the failing source line.
 *
 * Software rasterisation (SwiftShader) is deliberate and sufficient for the
 * question being asked: GLSL ES is validated by ANGLE's translator before it
 * ever reaches a driver, so a syntax or semantic error surfaces here exactly
 * as it would on a GPU. What this does NOT prove is visual correctness or
 * performance -- 8.88M splats through SwiftShader is not a frame rate, and
 * pixel output still needs a real GPU and a human.
 *
 * Usage (dev server must already be running on :5173):
 *   node test/gut-smoke.mjs
 */
import { chromium } from '@playwright/test'

const BASE = process.env.VIEWER_URL ?? 'http://localhost:5173'
const DEV_KEY = process.env.DEV_KEY ?? 'splat'
const SETTLE_MS = Number(process.env.SETTLE_MS ?? 90_000)

// Wraps the GL context in-page. Installed via addInitScript so it is in place
// before any module runs and cannot miss an early program build.
const INSTRUMENT = () => {
  const w = /** @type {any} */ (window)
  w.__gl = {
    shaders: [],
    programs: [],
    contexts: 0,
    compiledOk: 0,
    linkedOk: 0,
    draws: 0,
    gutShaderSeen: 0,
  }

  const wrap = (proto, label) => {
    if (!proto) return
    const origCompile = proto.compileShader
    const origLink = proto.linkProgram
    for (const fn of [
      'drawElements',
      'drawArrays',
      'drawElementsInstanced',
      'drawArraysInstanced',
    ]) {
      const orig = proto[fn]
      if (!orig) continue
      proto[fn] = function (...a) {
        w.__gl.draws++
        return orig.apply(this, a)
      }
    }
    proto.compileShader = function (shader) {
      origCompile.call(this, shader)
      // Does this shader actually carry the 3DGUT code? gutMode is a UNIFORM
      // and gutDefines.glsl is #included unconditionally, so the gut source is
      // in the compile path on every splat render no matter what the toggle
      // says. Counting it proves the check wasn't vacuous.
      const srcAll = this.getShaderSource(shader) || ''
      if (srcAll.includes('gutSplat') || srcAll.includes('uniform bool gutMode')) {
        w.__gl.gutShaderSeen++
      }
      const ok = this.getShaderParameter(shader, this.COMPILE_STATUS)
      if (ok) w.__gl.compiledOk++
      if (!ok) {
        const log = this.getShaderInfoLog(shader) || ''
        const src = this.getShaderSource(shader) || ''
        // Pull the line numbers the driver complained about so the report can
        // show the offending source rather than just the message.
        const lines = src.split('\n')
        const cited = [
          ...new Set(
            [...log.matchAll(/(?:^|\D)(\d+):(\d+)/g)].map((m) => Number(m[1])),
          ),
        ].filter((n) => n > 0 && n <= lines.length)
        w.__gl.shaders.push({
          api: label,
          type:
            this.getShaderParameter(shader, this.SHADER_TYPE) === this.VERTEX_SHADER
              ? 'vertex'
              : 'fragment',
          log,
          context: cited.map((n) => `${n}: ${lines[n - 1]}`),
        })
      }
      return undefined
    }
    proto.linkProgram = function (program) {
      origLink.call(this, program)
      const ok = this.getProgramParameter(program, this.LINK_STATUS)
      if (ok) w.__gl.linkedOk++
      if (!ok) {
        w.__gl.programs.push({ api: label, log: this.getProgramInfoLog(program) || '' })
      }
      return undefined
    }
  }
  wrap(w.WebGL2RenderingContext?.prototype, 'webgl2')
  wrap(w.WebGLRenderingContext?.prototype, 'webgl')

  const origGet = HTMLCanvasElement.prototype.getContext
  HTMLCanvasElement.prototype.getContext = function (type, attrs) {
    const ctx = origGet.call(this, type, attrs)
    if (ctx && String(type).startsWith('webgl')) w.__gl.contexts++
    return ctx
  }
}

async function run(browser, gut) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  await page.addInitScript(INSTRUMENT)

  const console_ = []
  const pageErrors = []
  const netFails = []
  page.on('console', (m) => console_.push({ type: m.type(), text: m.text() }))
  page.on('pageerror', (e) => pageErrors.push(String(e)))
  page.on('requestfailed', (r) =>
    netFails.push(`FAILED ${r.url()} :: ${r.failure()?.errorText ?? '?'}`),
  )
  page.on('response', (r) => {
    if (r.status() >= 400) netFails.push(`HTTP ${r.status()} ${r.url()}`)
  })

  const url = `${BASE}/?dev=${DEV_KEY}&gut=${gut}`
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 })

  // Wait for the canvas to actually show something, not merely for the shader
  // to compile. An earlier version of this stopped as soon as the first draw
  // call landed and reported a black screen as success: under
  // `frameloop: 'demand'` one draw proves only that a frame was requested,
  // and 257 MB of RAD chunks have not finished paging in by then.
  //
  // The signal used instead is non-black pixels in the canvas centre, sampled
  // off a downscaled copy. Bails early on a GL failure; falls through on
  // timeout so a hang still reports rather than hanging us.
  const deadline = Date.now() + SETTLE_MS
  let lit = 0
  for (;;) {
    const s = await page
      .evaluate(() => {
        const w = /** @type {any} */ (window)
        const out = {
          fail: w.__gl.shaders.length + w.__gl.programs.length,
          gut: w.__gl.gutShaderSeen,
          draws: w.__gl.draws,
          lit: 0,
        }
        // Largest canvas on the page == the R3F one.
        const cs = [...document.querySelectorAll('canvas')]
        const c = cs.sort((a, b) => b.width * b.height - a.width * a.height)[0]
        if (c && c.width) {
          const s2 = document.createElement('canvas')
          s2.width = 64
          s2.height = 64
          const cx = s2.getContext('2d')
          if (cx) {
            try {
              cx.drawImage(c, 0, 0, 64, 64)
              const d = cx.getImageData(0, 0, 64, 64).data
              for (let i = 0; i < d.length; i += 4) {
                if (d[i] > 8 || d[i + 1] > 8 || d[i + 2] > 8) out.lit++
              }
            } catch {
              /* tainted or context lost */
            }
          }
        }
        return out
      })
      .catch(() => null)
    if (!s) break
    lit = s.lit
    if (s.fail) break
    if (s.gut > 0 && lit > 40) break
    if (Date.now() >= deadline) break
    await page.waitForTimeout(2000)
  }

  const gl = await page.evaluate(() => /** @type {any} */ (window).__gl)
  const shot = `/tmp/gut-${gut}.png`
  await page.screenshot({ path: shot })
  await page.close()
  return { gut, gl, console_, pageErrors, netFails, lit, shot, url }
}

const browser = await chromium.launch({
  executablePath: '/usr/bin/google-chrome-stable',
  args: [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist',
  ],
})

const results = []
for (const gut of [0, 1]) results.push(await run(browser, gut))
await browser.close()

let bad = false
for (const r of results) {
  console.log(`\n${'='.repeat(70)}\n?gut=${r.gut}   ${r.url}\n${'='.repeat(70)}`)
  console.log(`gl contexts created   : ${r.gl.contexts}`)
  console.log(`shaders compiled OK   : ${r.gl.compiledOk}`)
  console.log(`shader compile FAILS  : ${r.gl.shaders.length}`)
  console.log(`programs linked OK    : ${r.gl.linkedOk}`)
  console.log(`program link FAILS    : ${r.gl.programs.length}`)
  console.log(`3DGUT shaders compiled: ${r.gl.gutShaderSeen}   <- 0 means this check proved NOTHING`)
  console.log(`draw calls            : ${r.gl.draws}`)
  console.log(`lit pixels /4096      : ${r.lit}   <- 0 means a BLACK canvas`)
  console.log(`screenshot            : ${r.shot}`)
  if (r.netFails.length) {
    console.log(`\nnetwork failures (${r.netFails.length}):`)
    for (const f of [...new Set(r.netFails)].slice(0, 12)) console.log('  ' + f)
  }
  if (r.gl.gutShaderSeen === 0) {
    bad = true
    console.log('  !! the splat shader never compiled -- asset never loaded, or the')
    console.log('     splat layer never mounted. Nothing about gutMode was tested.')
  }

  for (const s of r.gl.shaders) {
    bad = true
    console.log(`\n--- ${s.type} SHADER FAILED TO COMPILE (${s.api}) ---`)
    console.log(s.log.trim())
    if (s.context.length) console.log('offending source:\n  ' + s.context.join('\n  '))
  }
  for (const p of r.gl.programs) {
    bad = true
    console.log(`\n--- PROGRAM FAILED TO LINK (${p.api}) ---\n${p.log.trim()}`)
  }

  const errs = r.console_.filter((m) => m.type === 'error')
  if (errs.length) {
    console.log(`\nconsole errors (${errs.length}):`)
    for (const e of errs.slice(0, 12)) console.log('  ' + e.text.slice(0, 400))
  }
  const warns = r.console_.filter(
    (m) => m.type === 'warning' && /splat|spark|rad|page|gut/i.test(m.text),
  )
  if (warns.length) {
    console.log(`\nrelevant warnings (${warns.length}):`)
    for (const wn of warns.slice(0, 8)) console.log('  ' + wn.text.slice(0, 300))
  }
  if (r.pageErrors.length) {
    bad = true
    console.log(`\nuncaught page errors (${r.pageErrors.length}):`)
    for (const e of r.pageErrors.slice(0, 8)) console.log('  ' + e.slice(0, 400))
  }
}

console.log(`\n${'='.repeat(70)}`)
console.log(bad ? 'RESULT: FAILURES ABOVE' : 'RESULT: no shader/program/page errors in either mode')
process.exit(bad ? 1 : 0)
