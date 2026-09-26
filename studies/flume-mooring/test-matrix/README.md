# HSFP 1:50 wave test matrix: review and FloatSim predictions

| File | What it is |
|---|---|
| `HSFP_1-50_Wave_Test_Matrix_9-26-2026.xlsx` | The test matrix as received (2026-09-26), unchanged. |
| `HSFP_1-50_Wave_Test_Matrix_9-26-2026_FloatSim.xlsx` | The same workbook with FloatSim's pre-test predictions (Regular_Waves rows 24–26, and the FloatSim_Predictions sheet). All other cells are unchanged; the run counts and tank time are unchanged. |
| `fill_predictions.py` | Builds the `_FloatSim` workbook from the original and the study's records. |
| `TEST-MATRIX-REVIEW.md` | The review of the matrix against the mooring study, and what else the tests need. |

## Rebuilding

1. Run `python fill_predictions.py`.
2. Recalculate in Excel so the file carries computed values. Open and save it, or use PowerShell:

```
$xl = New-Object -ComObject Excel.Application; $xl.DisplayAlerts = $false
$wb = $xl.Workbooks.Open("<full path>\HSFP_1-50_Wave_Test_Matrix_9-26-2026_FloatSim.xlsx")
$xl.CalculateFull(); $wb.Save(); $wb.Close($false); $xl.Quit()
```

The predictions are pre-test values. Replace them with the measured free decays.
