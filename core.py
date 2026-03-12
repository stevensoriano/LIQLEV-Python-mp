# -*- coding: utf-8 -*-
"""
Core LIQLEV transient solver for parallel geometry sweep.
Adapted from the LIQLEV VBA model Python conversion.
"""
import numpy as np
import pandas as pd
from thermo_utils import Tsat, Psat, DensitySat, Cpsat, LHoV, dPdTsat, sli


def liqlev_simulation(inputs):
    """
    Run the LIQLEV transient boundary-layer simulation.

    Parameters
    ----------
    inputs : dict
        Simulation input dictionary from get_base_inputs().

    Returns
    -------
    pd.DataFrame
        Time-series results with columns: Time, Press, Height.
    """
    inputs = {k.lower(): v for k, v in inputs.items()}
    delta, units = inputs['delta'], inputs['units']
    nvmd, tvmdot, xvmdot = int(inputs['nvmd']), inputs['tvmdot'], inputs['xvmdot']
    neps, teps, xeps = int(inputs['neps']), inputs['teps'], inputs['xeps']
    tspal, xspacl = inputs['tspal'], inputs['xspacl']
    tspav, xspacv = inputs['tspav'], inputs['xspacv']
    tggo, xggo = inputs['tggo'], inputs['xggo']
    gravity_function = inputs.get('gravity_function', None)
    fluid = inputs['liquid']

    if units == "SI":
        dtank = inputs['dtank'] * 3.28
        htzero = inputs['htzero'] * 3.28
        volt = inputs['volt'] * (3.28 ** 3)
        xmlzro = inputs['xmlzro'] * 2.20462
        pinit = inputs['pinit'] * 0.145
        pfinal = inputs['pfinal'] * 0.145
        tinit = inputs['tinit'] * 1.8
    else:
        dtank = inputs['dtank']
        htzero = inputs['htzero']
        volt = inputs['volt']
        xmlzro = inputs['xmlzro']
        pinit = inputs['pinit']
        pfinal = inputs['pfinal']
        tinit = inputs['tinit']

    perim = np.pi * dtank
    ac = 0.7854 * (dtank ** 2)
    htank = volt / ac
    fill = htzero * ac / volt

    dhdt, vbl1, hldak3 = 0.0, 0.0, 0.0
    p1, t1, h1 = pinit, tinit, htzero
    theta1 = inputs['thetin']
    xml1, zht1 = xmlzro, htzero
    results = []
    xmvap1, xmvap2 = 0.0, 0.0
    is_first_iteration = True

    PSI_TO_KPA = 6.89475729

    while p1 > pfinal:
        if fluid == "Hydrogen":
            rhol = (0.1709 + 0.7454 * t1 - 0.04421 * t1**2
                    + 0.001248 * t1**3 - 1.738e-5 * t1**4 + 9.424e-8 * t1**5)
            rhov = (-0.2511 + 0.04294 * t1 - 0.00286 * t1**2
                    + 9.159e-5 * t1**3 - 1.422e-6 * t1**4 + 1.001e-8 * t1**5)
            cs = 0.078 * (t1 - 34.0) + 2.12
            hfg = -2.0 * (t1 - 34.0) + 194.5
            dpdts = 2.49 - 0.22 * t1 + 0.00407 * t1**2 + 5.22e-5 * t1**3
        else:
            t1_k = t1 / 1.8
            ps_kpa = Psat(fluid, t1_k)
            rhol = DensitySat(fluid, "liquid", ps_kpa) * 0.0624279606
            rhov = DensitySat(fluid, "vapor", ps_kpa) * 0.0624279606
            cs = Cpsat(fluid, "liquid", ps_kpa) * 0.2388458966
            hfg = LHoV(fluid, ps_kpa) * 0.4299226
            dpdts = dPdTsat(fluid, ps_kpa) * 0.08057652094

        volliq = xml1 / rhol + vbl1
        volgas = volt - volliq
        xmvap3 = volgas * rhov
        if is_first_iteration:
            xmvap1 = xmvap3

        dtdps = 1.0 / dpdts if dpdts != 0 else 0
        theta2 = theta1 + delta
        thetav = 0.5 * (theta1 + theta2)
        vmdot = sli(thetav, tvmdot, xvmdot)

        denom = (xml1 * cs * dtdps / hfg
                 + xmvap1 * (1 / p1 - dtdps / t1)) if hfg * p1 * t1 != 0 else 1
        dpdtha = -vmdot / denom if denom != 0 else 0
        delp = dpdtha * delta
        p2 = p1 + delp
        t2 = t1 + dtdps * delp
        delme = xml1 * cs * (t2 - t1) / hfg if hfg != 0 else 0
        xml2 = xml1 + delme
        delmv = vmdot * delta

        eps = (sli(thetav, teps, xeps) if neps > 0
               else (perim * h1) / (perim * h1 + ac))
        spacv = sli(thetav, tspav, xspacv)
        spacl = sli(thetav, tspal, xspacl)
        ggo_ft_s2 = (gravity_function(thetav) if gravity_function is not None
                     else sli(thetav, tggo, xggo))

        ak1_term = (10.8 * (1 + spacl) * (1 + spacv) * ggo_ft_s2
                     * (rhol - rhov) / rhol) if rhol != 0 else 0
        ak1 = 1.089 * (ak1_term ** 0.5) if ak1_term > 0 else 0
        ak2 = (-eps * cs * rhol * dtdps * dpdtha
                / rhov / hfg) if rhov * hfg != 0 else 0
        ak3 = ak2 / ak1 if ak1 != 0 else 0

        nconv = 0
        ak4, fvbl4 = 0, 0
        solver_loop_active = True
        while solver_loop_active and nconv < 80:
            zht2 = zht1 + dhdt * delta
            if ak3 < 0:
                ak3 = hldak3
            delblz = ((0.375 * dtank * ak3 * zht2) ** (2 / 3)
                      if (0.375 * dtank * ak3 * zht2) > 0 else 0)

            n_inner = 0
            while n_inner < 20:
                sum1 = sum(
                    (4 ** (l - 1)) * (delblz ** (l + 0.5))
                    / (dtank ** l) / (2 * l + 1)
                    for l in range(1, 11)
                )
                fdelt = 8.0 * sum1 / ak3 - zht2 if ak3 != 0 else float('inf')
                if abs(fdelt) <= 1e-5 * zht2:
                    break
                summ = sum(
                    (4 ** (k - 1)) * (delblz ** (k - 0.5))
                    / (dtank ** k)
                    for k in range(1, 11)
                )
                fpdelt = 4.0 * summ / ak3 if ak3 != 0 else float('inf')
                delblz -= fdelt / (fpdelt if fpdelt != 0 else 1e-9)
                n_inner += 1

            sum1_vbl = sum(
                (2 * l + 1) * (delblz ** (l + 1.5))
                / (l + 1.5) / (dtank ** (l - 1))
                for l in range(1, 11)
            )
            vbl2 = sum1_vbl * np.pi / ak3 if ak3 != 0 else 0
            fvbl = (vbl2 - ak2 * xml1 * delta / rhol
                    + 2.1 * ak1 * dtank * (delblz ** 1.5) * delta
                    - vbl1) if rhol != 0 else float('inf')

            if abs(fvbl) <= 0.001 * vbl2:
                solver_loop_active = False
            else:
                if nconv <= 1:
                    if nconv == 0:
                        nconv = 1
                    savak3, svfvbl = ak3, fvbl
                    if fvbl > 0:
                        ak3 *= 0.1
                    else:
                        ak3 *= 2.0
                        nconv = 0
                else:
                    if (fvbl > 0 and svfvbl < 0) or (fvbl < 0 and svfvbl > 0):
                        pass
                    else:
                        savak3, svfvbl = ak4, fvbl4
                    ak4, fvbl4 = ak3, fvbl
                    fvbl_diff = fvbl - svfvbl
                    secant_ak3 = (ak3 - fvbl * (ak3 - savak3) / fvbl_diff
                                  if fvbl_diff != 0 else ak3)
                    ak3 = 0.5 * (secant_ak3 + savak3)
                    nconv += 1

        dhdt = (((vbl2 - vbl1) + (xml2 - xml1) / rhol)
                / ac / delta) if rhol * ac * delta != 0 else 0
        zht2 = zht1 + dhdt * delta
        h2 = zht1 + dhdt * delta

        # Check overfill: clamp at tank height and terminate
        if h2 >= htank:
            h2 = htank
            results.append({'Time': theta2, 'Press': p2, 'Height': h2})
            break

        results.append({'Time': theta2, 'Press': p2, 'Height': h2})

        p1, t1, h1 = p2, t2, h2
        theta1, xml1, xmvap1 = theta2, xml2, xmvap2
        vbl1, zht1 = vbl2, zht2
        hldak3 = ak3
        is_first_iteration = False

        if theta2 >= tvmdot[-1] - delta:
            break

    return pd.DataFrame(results)
