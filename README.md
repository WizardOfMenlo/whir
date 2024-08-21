<h1 align="center">WHIR 🌪️</h1>

This library was developed using the [arkworks](https://arkworks.rs) ecosystem to accompany [WHIR 🌪️](https://eprint.iacr.org/2024/XXX). 
By [Gal Arnon](https://galarnon42.github.io/) [Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Giacomo Fenzi](https://gfenzi.io), and [Eylon Yogev](https://www.eylonyogev.com/about).

**WARNING:** This is an academic prototype and has not received careful code review. This implementation is NOT ready for production use.

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
- `-f` sets the field used, available are `Goldilocks2, Goldilocks3, Field192, Field256`.
- `--hash` sets the hash used for the Merkle tree, available are `SHA3` and `Blake3`
