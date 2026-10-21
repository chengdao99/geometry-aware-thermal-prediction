"""Parameterized building material library.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from . import material_config as cfg


@dataclass
class EnergyMaterialSpec:
    """Represent energy material spec and its associated geometry or physical configuration."""

    name: str
    conductivity: float
    density: float
    specific_heat: float
    roughness: str
    solar_absorptance: float = 0.7
    thermal_absorptance: float = 0.9
    visible_absorptance: float = 0.7


@dataclass
class LayerSpec:
    """Represent layer spec and its associated geometry or physical configuration."""

    material: EnergyMaterialSpec
    thickness: float

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return f"{self.material.name}_{int(round(self.thickness * 1000))}mm"

    @property
    def thermal_resistance(self) -> float:
        """Compute thermal resistance for the current building or model state."""
        return self.thickness / self.material.conductivity


@dataclass
class GlazingSpec:
    """Represent glazing spec and its associated geometry or physical configuration."""

    name: str
    thickness: float
    solar_transmittance: float
    solar_reflectance: float
    visible_transmittance: float
    visible_reflectance: float
    emissivity: float
    conductivity: float
    infrared_transmittance: float = 0.0
    solar_reflectance_back: Optional[float] = None
    emissivity_back: Optional[float] = None
    visible_reflectance_back: Optional[float] = None
    density: float = 2500.0
    specific_heat: float = 840.0

    def __post_init__(self):
        if self.solar_reflectance_back is None:
            self.solar_reflectance_back = self.solar_reflectance
        if self.emissivity_back is None:
            self.emissivity_back = self.emissivity
        if self.visible_reflectance_back is None:
            self.visible_reflectance_back = self.visible_reflectance

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return self.name


@dataclass
class ShadeMaterialSpec:
    """Represent shade material spec and its associated geometry or physical configuration."""

    name: str
    solar_transmittance: float
    solar_reflectance: float
    visible_transmittance: float
    visible_reflectance: float
    emissivity: float = 0.9
    thickness: float = 0.005
    conductivity: float = 0.05
    density: float = 1400.0
    specific_heat: float = 1300.0
    distance_to_glass: float = 0.05
    top_opening_multiplier: float = 0.5
    bottom_opening_multiplier: float = 0.5
    left_side_opening_multiplier: float = 0.5
    right_side_opening_multiplier: float = 0.5
    airflow_permeability: float = 0.0

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return f"ShadeMat__{self.name}"


@dataclass
class GasSpec:
    """Represent gas spec and its associated geometry or physical configuration."""

    name: str
    gas_type: str
    thickness: float
    conductivity: float = 0.0
    density: float = 0.0
    specific_heat: float = 0.0

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return self.name


@dataclass
class OpaqueConstructionSpec:
    """Represent opaque construction spec and its associated geometry or physical configuration."""

    layers: List[LayerSpec]

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return "OpaqueConst__" + "__".join((l.identifier for l in self.layers))

    @property
    def total_thickness(self) -> float:
        """Compute total thickness for the current building or model state."""
        return round(sum((l.thickness for l in self.layers)), 4)

    def u_value(self, r_si: float, r_se: float) -> float:
        """Compute u value for the current building or model state. Inputs: r_si, r_se."""
        r_layers = sum((l.thermal_resistance for l in self.layers))
        return round(1.0 / (r_si + r_layers + r_se), 4)

    @property
    def layer_identifiers(self) -> List[str]:
        """Compute layer identifiers for the current building or model state."""
        return [l.identifier for l in self.layers]


@dataclass
class WindowConstructionSpec:
    """Represent window construction spec and its associated geometry or physical configuration."""

    name: str
    layers: List[Union[GlazingSpec, GasSpec, ShadeMaterialSpec]]
    u_value: float
    shgc: float

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return f"WindowConst__{self.name}"


@dataclass
class AirBoundaryConstructionSpec:
    """Represent air boundary construction spec and its associated geometry or physical configuration."""

    name: str = "AirBoundary"

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return f"AirBoundaryConst__{self.name}"

    @property
    def total_thickness(self) -> float:
        """Compute total thickness for the current building or model state."""
        return 0.0

    @property
    def layers(self) -> list:
        """Compute layers for the current building or model state."""
        return []


@dataclass
class OutShadeConstructionSpec:
    """Represent out shade construction spec and its associated geometry or physical configuration."""

    name: str
    solar_reflectance: float
    visible_reflectance: float
    is_specular: bool = False
    conductivity: float = 160.0
    density: float = 2700.0
    specific_heat: float = 900.0

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return f"OutShadeConst__{self.name}"

    @property
    def solar_absorptance(self) -> float:
        """Compute solar absorptance for the current building or model state."""
        return round(1.0 - self.solar_reflectance, 4)


@dataclass
class InShadeConstructionSpec:
    """Represent in shade construction spec and its associated geometry or physical configuration."""

    name: str = "white_roller_blind"
    solar_reflectance: float = 0.5
    solar_transmittance: float = 0.3
    visible_reflectance: float = 0.55
    visible_transmittance: float = 0.35
    is_specular: bool = False
    conductivity: float = 0.05
    density: float = 1400.0
    specific_heat: float = 1300.0

    @property
    def identifier(self) -> str:
        """Compute identifier for the current building or model state."""
        return f"InShadeConst__{self.name}"

    @property
    def solar_absorptance(self) -> float:
        """Compute solar absorptance for the current building or model state."""
        return round(1.0 - self.solar_reflectance - self.solar_transmittance, 4)


CLT_WALL = EnergyMaterialSpec(
    name="CLT_Wall",
    conductivity=0.12,
    density=470,
    specific_heat=1600,
    roughness="MediumSmooth",
    solar_absorptance=0.65,
    thermal_absorptance=0.9,
    visible_absorptance=0.65,
)
CLT_SLAB = EnergyMaterialSpec(
    name="CLT_Slab",
    conductivity=0.12,
    density=470,
    specific_heat=1600,
    roughness="MediumSmooth",
    solar_absorptance=0.65,
    thermal_absorptance=0.9,
    visible_absorptance=0.65,
)
CONCRETE_WALL_NW = EnergyMaterialSpec(
    name="Concrete_NW_Wall",
    conductivity=1.74,
    density=2400,
    specific_heat=880,
    roughness="MediumRough",
)
CONCRETE_SLAB_NW = EnergyMaterialSpec(
    name="Concrete_NW_Slab",
    conductivity=1.5,
    density=2300,
    specific_heat=1000,
    roughness="MediumRough",
)
CONCRETE_WALL_LW = EnergyMaterialSpec(
    name="Concrete_LW_Wall",
    conductivity=0.8,
    density=1600,
    specific_heat=960,
    roughness="MediumRough",
)
CONCRETE_SLAB_LW = EnergyMaterialSpec(
    name="Concrete_LW_Slab",
    conductivity=0.8,
    density=1600,
    specific_heat=960,
    roughness="MediumRough",
)
LIGHTWOOD_WALL = EnergyMaterialSpec(
    name="LightWood_Wall",
    conductivity=0.13,
    density=500,
    specific_heat=1600,
    roughness="MediumSmooth",
    solar_absorptance=0.65,
    thermal_absorptance=0.9,
    visible_absorptance=0.65,
)
LIGHTWOOD_SLAB = EnergyMaterialSpec(
    name="LightWood_Slab",
    conductivity=0.13,
    density=500,
    specific_heat=1600,
    roughness="MediumSmooth",
    solar_absorptance=0.65,
    thermal_absorptance=0.9,
    visible_absorptance=0.65,
)
CONCRETE_WALL = CONCRETE_WALL_NW
CONCRETE_SLAB = CONCRETE_SLAB_NW
BRICK_WALL = EnergyMaterialSpec(
    name="Brick_Wall", conductivity=0.6, density=1800, specific_heat=840, roughness="Rough"
)
ROOF_MEMBRANE = EnergyMaterialSpec(
    name="Roof_Membrane",
    conductivity=0.16,
    density=1120,
    specific_heat=1460,
    roughness="MediumRough",
    solar_absorptance=0.65,
    thermal_absorptance=0.9,
    visible_absorptance=0.65,
)
AIR_BOUNDARY_MATERIAL = EnergyMaterialSpec(
    name="Air_Boundary",
    conductivity=100.0,
    density=1.2,
    specific_heat=1005,
    roughness="Smooth",
    solar_absorptance=0.0,
    thermal_absorptance=0.0,
    visible_absorptance=0.0,
)
EPS = EnergyMaterialSpec(
    name="EPS", conductivity=0.036, density=20, specific_heat=1450, roughness="MediumRough"
)
MINERAL_WOOL = EnergyMaterialSpec(
    name="MinWool", conductivity=0.035, density=30, specific_heat=840, roughness="MediumRough"
)
PIR = EnergyMaterialSpec(
    name="PIR", conductivity=0.022, density=35, specific_heat=1000, roughness="MediumRough"
)
STRUCTURAL_WALL_MATERIALS: dict[str, EnergyMaterialSpec] = {
    "CLT": CLT_WALL,
    "Concrete_NormalWeight": CONCRETE_WALL_NW,
    "Concrete_Lightweight": CONCRETE_WALL_LW,
    "Concrete": CONCRETE_WALL_NW,
    "Brick": BRICK_WALL,
    "LightWood": LIGHTWOOD_WALL,
}
STRUCTURAL_FLOOR_MATERIALS: dict[str, EnergyMaterialSpec] = {
    "CLT": CLT_SLAB,
    "Concrete_NormalWeight": CONCRETE_SLAB_NW,
    "Concrete_Lightweight": CONCRETE_SLAB_LW,
    "Concrete": CONCRETE_SLAB_NW,
    "Brick": CONCRETE_SLAB_NW,
    "LightWood": LIGHTWOOD_SLAB,
}
STRUCTURAL_ROOF_MATERIALS: dict[str, EnergyMaterialSpec] = {
    "CLT": CLT_SLAB,
    "Concrete_NormalWeight": CONCRETE_SLAB_NW,
    "Concrete_Lightweight": CONCRETE_SLAB_LW,
    "Concrete": CONCRETE_SLAB_NW,
    "Brick": CONCRETE_SLAB_NW,
    "LightWood": LIGHTWOOD_SLAB,
}
INSULATION_MATERIALS: dict[str, EnergyMaterialSpec] = {
    "EPS": EPS,
    "MinWool": MINERAL_WOOL,
    "PIR": PIR,
}
CLT_INTERIOR = EnergyMaterialSpec(
    name="CLT_Interior",
    conductivity=0.13,
    density=500.0,
    specific_heat=1600.0,
    roughness="MediumSmooth",
    solar_absorptance=0.5,
    thermal_absorptance=0.9,
    visible_absorptance=0.5,
)
CONCRETE_BLOCK_INTERIOR = EnergyMaterialSpec(
    name="Concrete_Block_Interior",
    conductivity=0.51,
    density=1400.0,
    specific_heat=840.0,
    roughness="MediumRough",
    solar_absorptance=0.65,
    thermal_absorptance=0.9,
    visible_absorptance=0.65,
)
BRICK_INTERIOR = EnergyMaterialSpec(
    name="Brick_Interior",
    conductivity=0.72,
    density=1600.0,
    specific_heat=840.0,
    roughness="MediumRough",
    solar_absorptance=0.65,
    thermal_absorptance=0.9,
    visible_absorptance=0.65,
)
GYPSUM_BOARD = EnergyMaterialSpec(
    name="Gypsum_Board",
    conductivity=0.16,
    density=784.0,
    specific_heat=830.0,
    roughness="MediumSmooth",
    solar_absorptance=0.6,
    thermal_absorptance=0.9,
    visible_absorptance=0.6,
)
INTERIOR_WALL_MATERIALS: dict[str, EnergyMaterialSpec] = {
    "CLT_Interior": CLT_INTERIOR,
    "Concrete_Block_Interior": CONCRETE_BLOCK_INTERIOR,
    "Brick_Interior": BRICK_INTERIOR,
    "Gypsum_Board": GYPSUM_BOARD,
}
INTERIOR_WALL_MAP: dict[str, str] = {
    "CLT": "CLT_Interior",
    "Concrete_NormalWeight": "Concrete_Block_Interior",
    "Concrete_Lightweight": "Concrete_Block_Interior",
    "Concrete": "Concrete_Block_Interior",
    "Brick": "Brick_Interior",
    "LightWood": "Gypsum_Board",
}
GLASS_CLEAR_6MM = GlazingSpec(
    name="Glass_Clear_6mm",
    thickness=0.006,
    solar_transmittance=0.77,
    solar_reflectance=0.07,
    visible_transmittance=0.88,
    visible_reflectance=0.08,
    emissivity=0.84,
    conductivity=1.0,
)
GLASS_LOW_E_6MM = GlazingSpec(
    name="Glass_LowE_6mm",
    thickness=0.006,
    solar_transmittance=0.6,
    solar_reflectance=0.11,
    visible_transmittance=0.75,
    visible_reflectance=0.12,
    emissivity=0.04,
    conductivity=1.0,
)
GLASS_LOW_E_4MM = GlazingSpec(
    name="Glass_LowE_4mm",
    thickness=0.004,
    solar_transmittance=0.6,
    solar_reflectance=0.11,
    visible_transmittance=0.75,
    visible_reflectance=0.12,
    emissivity=0.04,
    conductivity=1.0,
)
GLASS_SOLAR_CTRL_6MM = GlazingSpec(
    name="Glass_SolarCtrl_6mm",
    thickness=0.006,
    solar_transmittance=0.33,
    solar_reflectance=0.35,
    visible_transmittance=0.5,
    visible_reflectance=0.35,
    emissivity=0.2,
    conductivity=1.0,
)


def make_equivalent_shade_glazing(shade_spec: InShadeConstructionSpec) -> GlazingSpec:
    """Compute make equivalent shade glazing for the current building or model state. Inputs: shade_spec."""
    return GlazingSpec(
        name=f"{shade_spec.name}_layer",
        thickness=0.003,
        solar_transmittance=shade_spec.solar_transmittance,
        solar_reflectance=shade_spec.solar_reflectance,
        visible_transmittance=shade_spec.visible_transmittance,
        visible_reflectance=shade_spec.visible_reflectance,
        emissivity=0.9,
        conductivity=0.05,
        density=1400.0,
        specific_heat=1300.0,
    )


GAP_AIR_12MM = GasSpec(
    name="Air_12mm",
    gas_type="Air",
    thickness=0.012,
    conductivity=0.02551,
    density=1.277,
    specific_heat=1007.0,
)
GAP_ARGON_12MM = GasSpec(
    name="Argon_12mm",
    gas_type="Argon",
    thickness=0.012,
    conductivity=0.01772,
    density=1.699,
    specific_heat=519.0,
)
GAP_ARGON_16MM = GasSpec(
    name="Argon_16mm",
    gas_type="Argon",
    thickness=0.016,
    conductivity=0.01772,
    density=1.699,
    specific_heat=519.0,
)
GLAZING_SYSTEMS: dict[str, WindowConstructionSpec] = {
    "single_clear": WindowConstructionSpec(
        name="single_clear", layers=[GLASS_CLEAR_6MM], u_value=5.7, shgc=0.82
    ),
    "double_clear": WindowConstructionSpec(
        name="double_clear",
        layers=[GLASS_CLEAR_6MM, GAP_AIR_12MM, GLASS_CLEAR_6MM],
        u_value=2.8,
        shgc=0.76,
    ),
    "double_low_e_high_gain": WindowConstructionSpec(
        name="double_low_e_high_gain",
        layers=[GLASS_CLEAR_6MM, GAP_ARGON_12MM, GLASS_LOW_E_6MM],
        u_value=1.2,
        shgc=0.56,
    ),
    "double_low_e_solar_ctrl": WindowConstructionSpec(
        name="double_low_e_solar_ctrl",
        layers=[GLASS_SOLAR_CTRL_6MM, GAP_ARGON_12MM, GLASS_LOW_E_6MM],
        u_value=1.1,
        shgc=0.28,
    ),
    "triple_low_e_high_gain": WindowConstructionSpec(
        name="triple_low_e_high_gain",
        layers=[GLASS_CLEAR_6MM, GAP_ARGON_12MM, GLASS_LOW_E_6MM, GAP_ARGON_12MM, GLASS_LOW_E_4MM],
        u_value=0.7,
        shgc=0.42,
    ),
    "triple_low_e_solar_ctrl": WindowConstructionSpec(
        name="triple_low_e_solar_ctrl",
        layers=[
            GLASS_SOLAR_CTRL_6MM,
            GAP_ARGON_12MM,
            GLASS_LOW_E_6MM,
            GAP_ARGON_12MM,
            GLASS_LOW_E_4MM,
        ],
        u_value=0.65,
        shgc=0.2,
    ),
}
OUT_SHADE_CONSTRUCTIONS: dict[str, OutShadeConstructionSpec] = {
    "light_metal": OutShadeConstructionSpec(
        name="light_metal",
        solar_reflectance=0.55,
        visible_reflectance=0.6,
        conductivity=160.0,
        density=2700.0,
        specific_heat=900.0,
    ),
    "dark_metal": OutShadeConstructionSpec(
        name="dark_metal",
        solar_reflectance=0.25,
        visible_reflectance=0.2,
        conductivity=50.0,
        density=7800.0,
        specific_heat=500.0,
    ),
    "concrete": OutShadeConstructionSpec(
        name="concrete",
        solar_reflectance=0.4,
        visible_reflectance=0.4,
        conductivity=1.74,
        density=2400.0,
        specific_heat=880.0,
    ),
}
IN_SHADE_CONSTRUCTION: InShadeConstructionSpec = InShadeConstructionSpec()
IN_SHADE_MATERIAL = ShadeMaterialSpec(
    name="white_roller_blind",
    solar_transmittance=IN_SHADE_CONSTRUCTION.solar_transmittance,
    solar_reflectance=IN_SHADE_CONSTRUCTION.solar_reflectance,
    visible_transmittance=IN_SHADE_CONSTRUCTION.visible_transmittance,
    visible_reflectance=IN_SHADE_CONSTRUCTION.visible_reflectance,
)


def sample_structural_system(rng: random.Random, total_floor_area: Optional[float] = None) -> str:
    """Compute sample structural system for the current building or model state. Inputs: rng, total_floor_area."""
    if total_floor_area is not None and total_floor_area < cfg.STRUCTURAL_SYSTEM_AREA_THRESHOLD:
        weights_dict = cfg.STRUCTURAL_SYSTEM_WEIGHTS_SMALL
    else:
        weights_dict = cfg.STRUCTURAL_SYSTEM_WEIGHTS_LARGE
    keys = list(weights_dict.keys())
    weights = list(weights_dict.values())
    system = rng.choices(keys, weights=weights)[0]
    if system == "Concrete":
        sub_keys = list(cfg.CONCRETE_SUBTYPE_WEIGHTS.keys())
        sub_weights = list(cfg.CONCRETE_SUBTYPE_WEIGHTS.values())
        subtype = rng.choices(sub_keys, weights=sub_weights)[0]
        return f"Concrete_{subtype}"
    return system


def sample_insulation_key(rng: random.Random) -> str:
    """Compute sample insulation key for the current building or model state. Inputs: rng."""
    keys = list(cfg.INSULATION_WEIGHTS.keys())
    weights = list(cfg.INSULATION_WEIGHTS.values())
    return rng.choices(keys, weights=weights)[0]


def sample_glazing_system(rng: random.Random) -> WindowConstructionSpec:
    """Compute sample glazing system for the current building or model state. Inputs: rng."""
    keys = list(cfg.GLAZING_SYSTEM_WEIGHTS.keys())
    weights = list(cfg.GLAZING_SYSTEM_WEIGHTS.values())
    key = rng.choices(keys, weights=weights)[0]
    return GLAZING_SYSTEMS[key]


def sample_context_scenario(rng: random.Random) -> str:
    """Compute sample context scenario for the current building or model state. Inputs: rng."""
    keys = list(cfg.CONTEXT_SCENARIO_WEIGHTS.keys())
    weights = list(cfg.CONTEXT_SCENARIO_WEIGHTS.values())
    return rng.choices(keys, weights=weights)[0]


def sample_svf(rng: random.Random, scenario: str) -> Tuple[float, float]:
    """Compute sample svf for the current building or model state. Inputs: rng, scenario."""
    wall_lo, wall_hi = cfg.SVF_WALL_RANGES[scenario]
    roof_lo, roof_hi = cfg.SVF_ROOF_RANGES[scenario]
    svf_wall = round(rng.uniform(wall_lo, wall_hi), 2)
    svf_roof = round(rng.uniform(roof_lo, roof_hi), 2)
    return (svf_wall, svf_roof)


def make_air_boundary_construction() -> AirBoundaryConstructionSpec:
    """Compute make air boundary construction for the current building or model state."""
    return AirBoundaryConstructionSpec(name="AirBoundary")


def sample_out_shade_construction(rng: random.Random) -> OutShadeConstructionSpec:
    """Compute sample out shade construction for the current building or model state. Inputs: rng."""
    keys = list(cfg.OUT_SHADE_MATERIAL_WEIGHTS.keys())
    weights = list(cfg.OUT_SHADE_MATERIAL_WEIGHTS.values())
    return OUT_SHADE_CONSTRUCTIONS[rng.choices(keys, weights=weights)[0]]
