<h1 align="center">WHIR 🌪️</h1>

This library was developed using the [arkworks](https://arkworks.rs) ecosystem to accompany [WHIR 🌪️](https://eprint.iacr.org/2024/1586). 
By [Gal Arnon](https://galarnon42.github.io/) [Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Giacomo Fenzi](https://gfenzi.io), and [Eylon Yogev](https://www.eylonyogev.com/about).

**WARNING:** This is an academic prototype and has not received careful code review. This implementation is NOT ready for production use.

## Note on linear form binding

The WHIR and zkWHIR `prove`/`verify` entry points treat public linear forms
(weights) as caller-supplied inputs and do not absorb them into the
Fiat-Shamir transcript internally. Callers are expected to bind the forms
into the transcript themselves, matching whatever scheme they prefer.

This is required for soundness: without form binding, the verifier's only
check on each form is a single-point MLE equality at a form-independent
random point, which an adversarial prover can exploit by substituting an
alternate form that happens to agree at that point.

Any deterministic encoding works, as long as the prover and verifier
perform the *same* absorption in the *same* order. Common options:

- Absorb each form's defining data field-by-field as prover messages.
- Hash the forms and absorb the digest.
- Encode the forms via `DomainSeparator::instance(...)` when constructing
  the transcript.

Mismatched or omitted bindings will either reject honest proofs or weaken
soundness, so it is worth treating this step as part of the protocol setup
rather than an optional optimization.

<p align="center">
    <a href="https://github.com/WizardOfMenlo/whir/blob/main/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-APACHE-blue.svg"></a>
    <a href="https://github.com/WizardOfMenlo/whir/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

# Usage
```
cargo run --release -- --help

Usage: main [OPTIONS]

Options:
  -t, --type <PROTOCOL_TYPE>             [default: PCS]
  -l, --security-level <SECURITY_LEVEL>  [default: 100]
  -p, --pow-bits <POW_BITS>
  -d, --num-variables <NUM_VARIABLES>    [default: 20]
  -e, --evaluations <NUM_EVALUATIONS>    [default: 1]
  -r, --rate <RATE>                      [default: 1]
      --reps <VERIFIER_REPETITIONS>      [default: 1000]
  -k, --fold <FOLDING_FACTOR>            [default: 4]
      --sec <SOUNDNESS_TYPE>             [default: ConjectureList]
      --fold_type <FOLD_OPTIMISATION>    [default: ProverHelps]
  -f, --field <FIELD>                    [default: Goldilocks2]
      --hash <MERKLE_TREE>               [default: Blake3]
  -h, --help                             Print help
  -V, --version                          Print version
```

Options:
- `-t` can be either `PCS` or `LDT` to run as a (multilinear) PCS or a LDT
- `-l` sets the (overall) security level of the scheme
- `-p` sets the number of PoW bits (used for the query-phase). PoW bits for proximity gaps are set automatically.
- `-d` sets the number of variables of the scheme.
- `-e` sets the number of evaluations to prove. Only meaningful in PCS mode.
- `-r` sets the log_inv of the rate
- `-k` sets the number of variables to fold at each iteration. 
- `--sec` sets the settings used to compute security. Available `UniqueDecoding`, `ProvableList`, `ConjectureList`
- `--fold_type` sets the settings used to compute folds. Available `Naive`, `ProverHelps`
- `-f` sets the field used, available are `Goldilocks2, Goldilocks3, Field192, Field256`.
- `--hash` sets the hash used for the Merkle tree, available are `SHA3` and `Blake3`
