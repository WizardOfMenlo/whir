use serde::Serialize;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum WhirType {
    LDT,
    PCS,
}

impl FromStr for WhirType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "LDT" => Ok(Self::LDT),
            "PCS" => Ok(Self::PCS),
            _ => Err(format!("Invalid field: {s}")),
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
        match s {
            "Field128" => Ok(Self::Field128),
            "Field192" => Ok(Self::Field192),
            "Field256" => Ok(Self::Field256),
            "Goldilocks1" => Ok(Self::Goldilocks1),
            "Goldilocks2" => Ok(Self::Goldilocks2),
            "Goldilocks3" => Ok(Self::Goldilocks3),
            _ => Err(format!("Invalid field: {s}")),
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
        match s {
            "Keccak" => Ok(Self::Keccak256),
            "Blake3" => Ok(Self::Blake3),
            _ => Err(format!("Invalid hash: {s}")),
        }
    }
}
