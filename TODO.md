1. Investigate how to integrate the Camlp4 `kmodule` declaration into the PPX workflow:
   - extend `Sarek_ast`/`Sarek_typed_ast` with nodes that represent module constants and functions.
   - allow `[%kernel …]` payload to parse a `kmodule … end` block and register the contained definitions.
   - ensure the typer/lower/quote correctly see and reuse those definitions, then add a simple example and regression test showing shared constants/functions.
   - **Step 1 substeps**
     * Define `Sarek_ast.kmodule` and corresponding typed node to hold a list of constant/function declarations.
     * Update `Sarek_parse` to accept `module`/`let`-based declarations at the top of the kernel payload and emit `kmodule`.
     * Teach `Sarek_typer` to type-check the module items and register their definitions for later use in the kernel body.
     * Adjust `Sarek_lower`/`quote` to translate module constants/functions to `Kirc_Ast.GlobalFun`/`Intrinsics`.
     * Add a minimal example kernel plus regression test verifying the module is usable from the kernel body.

2. Add `ktype` declarations (records and variants) inside `%kernel` and propagate them through the PPX:
   - parse `ktype` as part of the kernel payload, supporting `mutable` fields and constructors.
   - teach the typer to understand these custom types and lower them to `Kirc_Ast` record/variant nodes for field access and pattern matching.
   - include an example kernel from `SpocLibs/Sarek/extension/ktype_examples.ml.reference` plus tests that verify IR matches the old Camlp4 output.

3. Introduce `klet` helper functions within the kernel payload:
   - allow `klet name = fun … -> body` inside `%kernel`, infer their types, and ensure other code can call them.
   - lower them to the existing `GlobalFun` machinery so the backend sees reusable helper kernels.
   - add an example/kernel that uses a helper and tests ensuring the helper is emitted and callable.

4. Expand the kernel grammar to cover the full `kern … -> …` syntax (arguments, pattern matching, modules/types):
   - parse the richer argument list (vector params, typed patterns, multiple statements) and reproduce the behavior of the Camlp4 `kern` rule.
   - make the PPX handle nested `module`/`type` definitions, `let` bindings, and pattern matches just like the old extension.
   - update tests/examples so kernels written with the full syntax compile through the PPX, comparing their IR to the Camlp4 references.

Each step should be implemented sequentially in its own commit with accompanying tests/examples.
