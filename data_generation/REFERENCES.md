# Generation references and assumptions

These sources provide background for building generation. Material values in [material_library.py](material_generation/material_library.py) are representative simulation inputs. Sampling intervals, probabilities, schedules and height multipliers in [material_config.py](material_generation/material_config.py) remain project assumptions unless a direct derivation is supplied. The illustrative passive scenario has zero equipment gains, occupant gains and active HVAC.

## Materials, glazing and shading

1. **ISO 10456:2007** — [Building-material hygrothermal properties](https://www.iso.org/standard/40966.html). Relevant to material thermal properties and design values. This scope check does not establish every library value. **Access:** official catalog and scope; complete standard requires separate access.

2. **EN 673:2011** — [Glazing thermal transmittance calculation](https://www.dinmedia.de/en/standard/din-en-673/138424268). Relevant to glazing U-values and insulating-gas heat transfer. **Access:** official catalog and scope for the historical edition; complete standard requires separate access.

3. **EN 410:2011** — [Luminous and solar characteristics of glazing](https://www.dinmedia.de/en/standard/din-en-410/138424295). Relevant to transmittance, reflectance and solar optical assessment methods. It does not validate every representative glazing system. **Access:** official catalog and scope; complete standard requires separate access.

4. **Pilkington Spectrum** — [Manufacturer's glazing performance tool](https://www.pilkington.com/en/my/commercial-applications/specification-tools/pilkington-spectrum?w=1200). Covers optical properties under EN 410 and centre-pane U-values under EN 673. Code values have not been matched to a specific product export. **Access:** public manufacturer tool description; not an archived product-data export.

5. **ISO 6946:2017** — [Thermal resistance and transmittance of building components](https://committee.iso.org/cms/live/live/en/sites/isoorg/contents/data/standard/06/57/65708.html?browse=ics). Relevant to opaque-component resistance calculations and internal/external surface resistances. **Access:** official catalog and scope; complete standard requires separate access.

6. **EN 14501:2021** — [Shading thermal and visual performance classification](https://www.dinmedia.de/en/standard/din-en-14501/311958504). Relevant to solar protection and shading optical performance. It does not establish the selected polyester density, specific heat or every fixed reflectance. **Access:** official catalog and scope for the historical edition; complete standard requires separate access.

7. **ASHRAE, 2021 Handbook—Fundamentals** — [Official chapter list](https://www.ashrae.org/technical-resources/publications-library/ashrae-handbook/table-of-contents-2021-ashrae-handbook-fundamentals). Chapters 18, 26 and 15 cover load calculations, material properties and fenestration respectively. **Access:** public official contents; the link is not the entire handbook.

## Occupants and operation

8. **ANSI/ASHRAE Standard 55-2020** — [Official Addendum b, approved in 2021](https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/55_2020_b_20210430.pdf). Confirms the historical standard and discusses metabolic-rate limits within thermal-comfort assessment. The code's 120 W/person remains a representative choice whose exact derivation was not verified. **Access:** public official addendum PDF, not the complete standard.

9. **ANSI/ASHRAE Standard 62.1-2022** — [Official interpretation record](https://www.ashrae.org/technical-resources/standards-and-guidelines/standards-interpretations/interpretations-for-standard-62-1-2022) and [official Addendum ab](https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/62_1_2022_ab_20231031.pdf). Table 6-1 provides ventilation/default occupancy information by space category. This is occupancy context, not a Swedish residential sampling distribution. **Access:** public official records and addendum PDF, not the complete standard.

10. **ISO 7730:2005** — [PMV/PPD thermal-comfort assessment](https://www.iso.org/standard/39155.html). Relevant to comfort/activity context. The occupant radiant fraction of 0.30 remains a project assumption. **Access:** official catalog and scope for the historical edition; complete standard requires separate access.

11. **Artmann, Manz and Heiselberg (2007).** [Climatic potential for passive cooling of buildings by night-time ventilation in Europe](https://vbn.aau.dk/en/publications/climatic-potential-for-passive-cooling-of-buildings-by-night-time/). *Applied Energy*, 84(2), 187–201. DOI: `10.1016/j.apenergy.2006.05.004`. Supports the climatic rationale for night cooling in Europe, without establishing the generator's ACH ranges/probabilities. **Access:** public university record and abstract. The journal issue is from 2007; the university record uses the online-publication date, 24 August 2006.

12. **Fabi, Andersen, Corgnati and Olesen (2013).** [A methodology for modelling energy-related human behaviour: Application to window opening behaviour in residential buildings](https://link.springer.com/article/10.1007/s12273-013-0119-6). *Building Simulation*, 6, 415–427. DOI: `10.1007/s12273-013-0119-6`. Relevant to probabilistic window-operation modelling, without establishing the generator's specific daily templates/time boundaries. **Access:** publisher metadata and abstract; full article may require subscription.

13. **Wilkins and Hosni (2000).** *Heat gain from office equipment*. *ASHRAE Journal*, 42(6), 33–44, as recorded in [ASHRAE's own handbook references](https://handbook.ashrae.org/Handbooks/F21/SI/F21_Ch18/F21_Ch18_si.aspx). Supports equipment heat-gain context. Generator intensity ranges and a uniform radiant fraction remain representative assumptions. **Access:** public publisher citation record within the handbook chapter; this is not a direct full-text link to the journal article.

## Urban radiation

14. **Lindberg, Holmer and Thorsson (2008).** [SOLWEIG 1.0 — Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings](https://link.springer.com/article/10.1007/s00484-008-0162-7). *International Journal of Biometeorology*, 52, 697–713. DOI: `10.1007/s00484-008-0162-7`. Relevant to urban geometry/radiation and Gothenburg measurements. The generator's SVF intervals, weights and floor-height multipliers were not derived or validated from this paper. **Access:** publisher metadata and abstract; full article may require subscription.
