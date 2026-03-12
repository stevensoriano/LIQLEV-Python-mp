# -*- coding: utf-8 -*-
"""
Thermodynamic utility functions for LIQLEV-Python-parallel.
CoolProp wrappers for cryogenic fluid saturation properties.
"""
import numpy as np
from CoolProp.CoolProp import PropsSI


def Tsat(fluid, press_kpa):
    """Saturation temperature (K) from pressure (kPa)."""
    return PropsSI("T", "P", press_kpa * 1000, "Q", 1, fluid)


def Psat(fluid, temp_sat_k):
    """Saturation pressure (kPa) from temperature (K)."""
    return PropsSI("P", "T", temp_sat_k, "Q", 1, fluid) / 1000


def DensitySat(fluid, phase, press_kpa):
    """Saturation density (kg/m^3). phase: 'liquid' or 'vapor'."""
    quality = 1 if phase == "vapor" else 0
    return PropsSI("Dmass", "P", press_kpa * 1000, "Q", quality, fluid)


def Cpsat(fluid, phase, press_kpa):
    """Specific heat at saturation (kJ/kg-K). phase: 'liquid' or 'vapor'."""
    quality = 1 if phase == "vapor" else 0
    return PropsSI("C", "P", press_kpa * 1000, "Q", quality, fluid) / 1000


def EnthalpySat(fluid, phase, press_kpa):
    """Saturation enthalpy (kJ/kg). phase: 'liquid' or 'vapor'."""
    quality = 1 if phase == "vapor" else 0
    return PropsSI("H", "P", press_kpa * 1000, "Q", quality, fluid) / 1000


def LHoV(fluid, press_kpa):
    """Latent heat of vaporization (kJ/kg)."""
    return EnthalpySat(fluid, "vapor", press_kpa) - EnthalpySat(fluid, "liquid", press_kpa)


def dPdTsat(fluid, press_kpa):
    """Slope of saturation curve (kPa/K) via finite difference."""
    t_plus = Tsat(fluid, press_kpa + 0.1)
    t_minus = Tsat(fluid, press_kpa - 0.1)
    if t_plus == t_minus:
        return 0
    return 0.2 / (t_plus - t_minus)


def sli(arg, x, y):
    """Simple linear interpolation (wraps numpy.interp)."""
    return np.interp(arg, x, y)
