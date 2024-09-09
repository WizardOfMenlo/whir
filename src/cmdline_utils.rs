use std::str::FromStr;

use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum WhirType {
    LDT,
    PCS,
}

impl FromStr for WhirType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "LDT" {
            Ok(Self::LDT)
        } else if s == "PCS" {
            Ok(Self::PCS)
        } else {
            Err(format!("Invalid field: {}", s))
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum AvailableFields {
    Goldilocks1, // Just Goldilocks
    Goldilocks2, // Quadratic extension of Goldilocks
    Goldilocks3, // Cubic extension of Goldilocks
    Field128,    // 128-bit prime field
    Field192,    // 192-bit prime field
    Field256,    // 256-bit prime field
}

impl FromStr for AvailableFields {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "Field128" {
            Ok(Self::Field128)
        } else if s == "Field192" {
            Ok(Self::Field192)
        } else if s == "Field256" {
            Ok(Self::Field256)
        } else if s == "Goldilocks1" {
            Ok(Self::Goldilocks1)
        } else if s == "Goldilocks2" {
            Ok(Self::Goldilocks2)
        } else if s == "Goldilocks3" {
            Ok(Self::Goldilocks3)
        } else {
            Err(format!("Invalid field: {}", s))
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum AvailableMerkle {
    Keccak256,
    Blake3,
}

impl FromStr for AvailableMerkle {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "Keccak" {
            Ok(Self::Keccak256)
        } else if s == "Blake3" {
            Ok(Self::Blake3)
        } else {
            Err(format!("Invalid hash: {}", s))
        }
    }
}
