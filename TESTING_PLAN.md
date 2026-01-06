# Priority 15: Unit Test Coverage for PPX Modules

## Current Status

**Coverage: 13/29 modules (~45%)**

### ✅ Already Tested (13 modules)
- Sarek_core_primitives (382 lines of tests)
- Sarek_env (324 lines)
- Sarek_lower (443 lines)
- Sarek_mono (182 lines)
- Sarek_parse (559 lines)
- Sarek_scheme (165 lines)
- Sarek_tailrec (265 lines)
- Sarek_typer (727 lines)
- Sarek_types (220 lines)
- Plus: float32/64, fusion, interp

### ❌ Untested Modules (16 modules)

#### **High Priority** (Core Logic)
1. **Sarek_error** - Error type definitions
   - Test: Error constructors, to_string, location tracking
   
2. **Sarek_lower_ir** - IR lowering from typed AST to Kirc
   - Test: Expression lowering, kernel conversion, type mapping
   
3. **Sarek_quote** - AST quoting for code generation
   - Test: quote_kernel, quote_sarek_expr, intrinsic collection
   
4. **Sarek_tailrec_analysis** - NEW: Recursion analysis (233 lines)
   - Test: count_recursive_calls, is_tail_recursive, analyze_recursion
   
5. **Sarek_tailrec_elim** - NEW: Tail recursion elimination (365 lines)
   - Test: eliminate_tail_recursion, loop generation
   
6. **Sarek_tailrec_pragma** - NEW: Pragma-based inlining (373 lines)
   - Test: parse_sarek_inline_pragma, inline_with_pragma, node counting

7. **Sarek_native_helpers** - NEW: Native codegen helpers (118 lines)
   - Test: default_value_for_type, identifier creation

8. **Sarek_native_intrinsics** - NEW: Intrinsic mapping (240 lines)
   - Test: gen_intrinsic_const, gen_intrinsic_fun, type mapping

#### **Medium Priority** (Infrastructure)
9. **Sarek_ppx_registry** - Type/intrinsic registration
   - Test: Registry operations, lookup functions

10. **Sarek_quote_ir** - IR quoting
    - Test: Kernel IR quoting

11. **Sarek_ir_ppx** - IR processing
    - Test: IR transformations

#### **Low Priority** (Support/Data)
12. **Sarek_ast** - AST type definitions (mostly data)
13. **Sarek_typed_ast** - Typed AST definitions (mostly data)
14. **Kirc_Ast** - Kirc IR definitions (mostly data)
15. **Sarek_reserved** - Reserved word checking
16. **Sarek_debug** - Debug logging
17. **Sarek_convergence** - Type convergence
18. **Sarek_tailrec_bounded** - Bounded inlining (92 lines, currently unused)

## Proposed Testing Strategy

### Phase 1: New Modules from Recent Splits (HIGH PRIORITY)
Test the 4 newly split tailrec modules and 2 native helper modules since they're fresh in mind:

1. **test_tailrec_analysis.ml** (~150 lines)
   - Test count_recursive_calls with various expression types
   - Test is_tail_recursive detection (tail vs non-tail cases)
   - Test has_recursion_in_loops
   - Test analyze_recursion full workflow

2. **test_tailrec_elim.ml** (~100 lines)
   - Test eliminate_tail_recursion produces valid loop structure
   - Test parameter substitution
   - Test continue/result variable generation

3. **test_native_helpers.ml** (~80 lines)
   - Test default_value_for_type for all types
   - Test identifier helper functions

4. **test_native_intrinsics.ml** (~120 lines)
   - Test gen_intrinsic_const with different modes
   - Test gen_intrinsic_fun with various intrinsics
   - Test type mapping functions

### Phase 2: Core Untested Logic (MEDIUM PRIORITY)
5. **test_error.ml** (~60 lines)
   - Test error constructors and messages
   
6. **test_quote.ml** (~200 lines)
   - Test kernel quoting
   - Test intrinsic collection

7. **test_lower_ir.ml** (~150 lines)
   - Test expression lowering
   - Test kernel conversion

### Phase 3: Infrastructure (LOWER PRIORITY)
8. test_ppx_registry.ml
9. test_quote_ir.ml
10. test_reserved.ml

### Phase 4: Data-heavy modules (OPTIONAL)
- AST definitions mostly don't need tests
- Already validated by integration tests

## Testing Framework

**Using Alcotest** (already in use):
```ocaml
open Sarek_ppx_lib.Sarek_tailrec_analysis

let test_count_no_recursion () =
  let expr = mk_texpr (TEInt 42) t_int32 in
  let count = count_recursive_calls "f" expr in
  Alcotest.(check int) "no calls" 0 count

let () =
  Alcotest.run "Sarek_tailrec_analysis" [
    ("count_recursive_calls", [
      Alcotest.test_case "no recursion" `Quick test_count_no_recursion;
      ...
    ]);
  ]
```

## Benefits

1. **Catch Regressions**: Especially after module splits
2. **Document Behavior**: Tests serve as examples
3. **Enable Refactoring**: Safe to change with test coverage
4. **Build Confidence**: Verify edge cases and invariants

## Incremental Approach

- Start with Phase 1 (new modules) - **~450 lines of tests**
- Each module gets its own test file
- Run `make test-all` after each addition
- Commit working tests incrementally

## Success Metrics

- **Short term**: Add 6 new test files (Phase 1+2) - coverage to ~65%
- **Medium term**: Add remaining tests - coverage to ~80%
- **Each test file**: 50-200 lines, focused on public API

---

**Ready to start with Phase 1?** Begin with test_tailrec_analysis.ml since that module is fresh?
