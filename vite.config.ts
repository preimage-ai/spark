import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import arraybuffer from "vite-plugin-arraybuffer";
import glsl from "vite-plugin-glsl";

const sparkRsDirectory = "rust/spark-rs/pkg";
if (!fs.existsSync(sparkRsDirectory) && !process.env.VITEST) {
  console.error(
    "\x1b[31m************************************************************************\x1b[0m",
  );
  console.error(
    "\x1b[31m Rust Wasm component not found, make sure to build them first.\x1b[0m",
  );
  console.error(
    "\x1b[31m Install Rust and run:\x1b[1m npm run build:wasm\x1b[0m",
  );
  console.error(
    "\x1b[31m************************************************************************\x1b[0m",
  );
  process.exit(1);
}

const assetsDirectory = "examples/assets";
const localAssetsDirectoryExist = fs.existsSync(assetsDirectory);
if (!localAssetsDirectoryExist) {
  console.log(
    "************************************************************************",
  );
  console.log(" Examples assets will be fetched from an external server.");
  console.log(
    " To work offline you can download them: npm run assets:download",
  );
  console.log(
    "************************************************************************",
  );
}

export default defineConfig(({ mode }) => {
  const isMinify = mode === "production";
  const isFirstPass = mode === "production";

  return {
    appType: "mpa",

    // Resolve `spark-rs` to the Rust build output directly, NOT through
    // node_modules. This is a correctness fix, not a convenience.
    //
    // `spark-rs` is declared as `file:rust/spark-rs/pkg`, which bun materialises
    // as a separate copy under node_modules/.bun/. That copy starts life
    // hard-linked to pkg/, so the two agree - until `build:wasm` runs:
    //
    //   1. wasm-pack writes its wasm-bindgen output INTO the shared inode, so
    //      node_modules sees it too.
    //   2. wasm-opt then replaces pkg/spark_rs_bg.wasm with a NEW file,
    //      breaking the hard link.
    //
    // pkg/ ends up with the optimized wasm; node_modules stays frozen on the
    // pre-wasm-opt intermediate (~1.93 MB vs ~1.67 MB). Since src/wasm.ts
    // imports the binary by package specifier, the build inlined that
    // intermediate - a ~350 kB regression that every timestamp reported as
    // fresh, and that `npm run build:wasm && npm run build` could not fix
    // because the second half never saw the first half's real output.
    //
    // Worse, the stale copy survives a package rename: an orphaned
    // node_modules/.bun entry for the OLD directory name kept serving a wasm
    // from weeks earlier, which is how a release shipped without the
    // clustered-SH fix that was sitting in the source tree.
    //
    // Pointing at pkg/ makes build:wasm -> build deterministic and removes
    // node_modules from the wasm's path entirely. The regex keeps the bare
    // specifier and subpath+query imports (`spark-rs/spark_rs_bg.wasm?...`)
    // both resolving to the same place, so the JS glue and the binary can
    // never again come from different builds.
    resolve: {
      alias: [
        {
          find: /^spark-rs(\/.*)?$/,
          replacement: `${path.resolve(import.meta.dirname, "rust/spark-rs/pkg")}$1`,
        },
      ],
    },

    plugins: [
      arraybuffer(),
      glsl({
        include: ["**/*.glsl"],
      }),

      {
        name: "serve-node-modules-alias",
        configureServer(server) {
          const baseUrlPath = "/examples/js/vendor/";

          server.middlewares.use((req, res, next) => {
            if (!req.url?.startsWith(baseUrlPath)) return next();

            const relModulePath = req.url.slice(baseUrlPath.length); // safe substring
            const absPath = path.resolve("node_modules", relModulePath);

            if (fs.existsSync(absPath) && fs.statSync(absPath).isFile()) {
              const ext = path.extname(absPath);
              const contentType =
                {
                  ".js": "application/javascript",
                  ".mjs": "application/javascript",
                  ".css": "text/css",
                  ".json": "application/json",
                }[ext] || "application/octet-stream";

              res.setHeader("Content-Type", contentType);
              fs.createReadStream(absPath).pipe(res);
            } else {
              res.statusCode = 404;
              res.end(`Not found: ${relModulePath}`);
            }
          });

          console.log(`📦 Dev alias active: ${baseUrlPath} → node_modules/*`);
        },
      },
    ],

    build: {
      minify: isMinify,
      lib: {
        entry: path.resolve(import.meta.dirname, "src/index.ts"),
        name: "spark",
        formats: ["es", "cjs"],
        fileName: (format) => {
          const base = format === "es" ? "spark.module" : `spark.${format}`;
          return isMinify ? `${base}.min.js` : `${base}.js`;
        },
      },
      sourcemap: true,
      rollupOptions: {
        external: ["three", /^three\/addons/],
        output: {
          globals: {
            three: "THREE",
          },
        },
      },
      emptyOutDir: isFirstPass,
    },

    worker: {
      rollupOptions: {
        treeshake: true,
      },
      plugins: () => [
        glsl({
          include: ["**/*.glsl"],
        }),
      ],
    },

    server: {
      watch: {
        usePolling: true,
      },
      port: 8080,
    },

    optimizeDeps: {
      force: true,
      exclude: ["three"], // prevent Vite pre-bundling
    },

    // Expose SPARK_* env vars on import.meta.env (e.g. SPARK_ENABLE_HOOKS for
    // the test-only await points in src/hooks.ts). Unset vars are absent, so
    // checks against them fold to false in normal builds.
    envPrefix: ["VITE_", "SPARK_"],

    define: {
      sparkLocalAssets: localAssetsDirectoryExist,
    },

    test: {
      setupFiles: ["test/unit/setup.ts"],
      coverage: {
        provider: "v8",
      },
      restoreMocks: true,
    },
  };
});
