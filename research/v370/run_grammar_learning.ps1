
param(
    [int]$TrainLimit = 0,
    [int]$Heldout = 5000
)

$corpus = ".\data\BabyLM-2026-Strict-Small"
$conceptnet = ".\data\conceptnet_compact.db"

if ($TrainLimit -gt 0) {
    python .\research\v369\grammar_experiment.py `
      $corpus `
      --conceptnet $conceptnet `
      --train-limit $TrainLimit `
      --heldout $Heldout
} else {
    python .\research\v369\grammar_experiment.py `
      $corpus `
      --conceptnet $conceptnet `
      --heldout $Heldout
}
