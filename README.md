<h1 align="center">WHIR 🌪️</h1>

This library was developed using the [arkworks](https://arkworks.rs) ecosystem to accompany [WHIR 🌪️](https://eprint.iacr.org/2024/1586). 
By [Gal Arnon](https://galarnon42.github.io/) [Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Giacomo Fenzi](https://gfenzi.io), and [Eylon Yogev](https://www.eylonyogev.com/about).

**WARNING:** This is an academic prototype and has not received careful code review. This implementation is NOT ready for production use.

## ⚠️ Soundness — caller must bind public linear forms into the transcript

The WHIR and zkWHIR `prove`/`verify` entry points **do not** absorb the public
linear forms (a.k.a. weights) into the Fiat-Shamir transcript. **Binding the
forms is the caller's responsibility**, and forgetting to do so breaks
soundness.

Without form binding, a malicious prover can take an honest proof for
`⟨w, f⟩ = e` and present it to the verifier under an alternate form `w'`
whose multilinear extension agrees with `w` at the final sumcheck point.
The verifier accepts the false claim `⟨w', f⟩ = e` because the only check
on the form is a single-point MLE equality — and that point is
form-independent unless the form is bound in the transcript.

### What the caller must do

Before calling `prove`, the caller must absorb the linear forms into the
prover's transcript. Before calling `verify`, the caller must mirror the
same absorption (reading and checking against the verifier's local copy)
in the verifier's transcript. Any binding scheme is acceptable as long as
it uniquely determines the forms in the transcript:

- Absorb each form's defining data field-by-field as prover messages.
- Hash the forms and absorb the digest.
- Encode the forms into `DomainSeparator::instance(...)` before
  constructing the transcript.
- Or any other deterministic encoding the prover and verifier agree on.

The prover and verifier **must** use the exact same binding sequence.
Failure to bind, or mismatched bindings between the two sides, breaks
soundness or rejects honest proofs.

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
