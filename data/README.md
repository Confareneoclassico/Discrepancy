# HYMOD data

Fifty thousand runs of the HYMOD conceptual rainfall-runoff model
(Vrugt et al., 2003) for the Leaf River catchment (Mississippi).
Simulation period: 28 July 1952 to 30 September 1962 (about ten years and two
months).

The design is an Owen-scrambled Sobol' sequence (`qrng::sobol`,
`randomize = "Owen"`, `seed = 123`) mapped to uniform distributions.

| Column | Parameter | Distribution (`hydro2.csv`) | Description |
|--------|-----------|-----------------------------|-------------|
| x1 | Cmax  | U(1, 500)    | maximum storage capacity of the catchment [mm] |
| x2 | bexp  | U(0.1, 2)    | spatial variability of the soil moisture capacity [-] |
| x3 | alpha | U(0.1, 0.99) | factor distributing the flow between the quick- and slow-flow reservoirs [-] |
| x4 | Rs    | U(0, 0.1)    | recession (residence-time) parameter of the slow-flow reservoir [-] |
| x5 | Rq    | U(0.1, 0.99) | recession (residence-time) parameter of the quick-flow reservoirs [-] |

- `hydro.csv`: the design as first generated, with x3 and x5 on [0.1, 0.98].
- `hydro2.csv`: the inputs as run (x3 and x5 rescaled to [0.1, 0.99]) and the
  output column `NSE`.

The code recovers the uniform [0, 1] design from `hydro.csv` and reads the
output from `hydro2.csv`. Version 1.x of the code and the first version of the
paper labelled x4 as Rq and x5 as Rs; the ranges show that x4 is Rs and x5 is
Rq.
