param(
  [Parameter(Mandatory = $false)]
  [string]$PythonExe = "C:\Users\60585\miniconda3\envs\quantEnv\python.exe",

  [Parameter(Mandatory = $false)]
  [string]$RepoRoot = "",

  [Parameter(Mandatory = $false)]
  [string]$MarketStatePath = "",

  [Parameter(Mandatory = $false)]
  [string]$MarketDaySummaryPath = "",

  [Parameter(Mandatory = $false)]
  [int]$Seed = 15,

  [Parameter(Mandatory = $false)]
  [string]$RunTag = (Get-Date -Format "yyyyMMdd_HHmmss"),

  [Parameter(Mandatory = $false)]
  [switch]$Execute,

  [Parameter(Mandatory = $false)]
  [switch]$SkipCompare
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($MarketStatePath)) {
  $MarketStatePath = Join-Path $RepoRoot "artifacts/market_state/daily_market_field_csi300.pkl"
}

if ([string]::IsNullOrWhiteSpace($MarketDaySummaryPath)) {
  $MarketDaySummaryPath = Join-Path $RepoRoot "artifacts/market_state/daily_market_observation_csi300.pkl"
}

if (-not (Test-Path $PythonExe)) {
  throw "Python interpreter not found: $PythonExe"
}

if (-not (Test-Path $MarketStatePath)) {
  throw "Market state asset not found: $MarketStatePath"
}

if (-not (Test-Path $MarketDaySummaryPath)) {
  throw "Market day summary asset not found: $MarketDaySummaryPath"
}

$RunLogRoot = Join-Path $RepoRoot "artifacts/experiment_logs/hierarchical_state_field/$RunTag"
$CompareOutRoot = Join-Path $RepoRoot "next_step/hierarchical_state_field/runs/$RunTag"

New-Item -ItemType Directory -Force -Path $RunLogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CompareOutRoot | Out-Null

function Quote-Token {
  param([Parameter(Mandatory = $true)][string]$Token)
  if ($Token -notmatch '\s') {
    return $Token
  }
  return '"' + $Token.Replace('"', '\"') + '"'
}

function Format-Command {
  param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string[]]$Args
  )

  return ((@(Quote-Token $Exe) + ($Args | ForEach-Object { Quote-Token $_ })) -join " ")
}

function New-Variant {
  param(
    [Parameter(Mandatory = $true)][int]$Order,
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][string]$Description,
    [Parameter(Mandatory = $true)][string[]]$Args
  )

  return [ordered]@{
    Order = $Order
    Label = $Label
    Description = $Description
    Args = $Args
  }
}

$CommonHierArgs = @(
  "--market_state_path", $MarketStatePath,
  "--market_day_summary_path", $MarketDaySummaryPath,
  "--experiment_suffix", "hsf_ablation_$RunTag",
  "--seed", "$Seed",
  "--use_hierarchical_state_field", "1",
  "--d_global_state", "64",
  "--d_local_state", "64",
  "--global_state_use_macro", "1",
  "--global_state_use_day_summary", "1",
  "--local_state_input_mode", "last_mean_std_trend_vol"
)

$Variants = @(
  (New-Variant -Order 1 -Label "global_local_stable_summary" -Description "Target configuration: hierarchical global+local with stable day_asset summary." -Args (
      $CommonHierArgs + @(
        "--router_summary_source", "day_asset",
        "--pooling_summary_source", "day_asset",
        "--label", "global_local_stable_summary",
        "--router_use_global_state", "1",
        "--router_use_local_state", "1",
        "--film_use_global_state", "1",
        "--film_use_local_state", "1",
        "--pooling_use_global_state", "1",
        "--pooling_use_local_state", "1"
      )
    )),
  (New-Variant -Order 2 -Label "base" -Description "Legacy control with day_asset summary and no hierarchical state field." -Args @(
      "--market_state_path", $MarketStatePath,
      "--market_day_summary_path", $MarketDaySummaryPath,
      "--router_summary_source", "day_asset",
      "--pooling_summary_source", "day_asset",
      "--label", "base",
      "--experiment_suffix", "hsf_ablation_$RunTag",
      "--seed", "$Seed",
      "--use_hierarchical_state_field", "0"
    )),
  (New-Variant -Order 3 -Label "global_local" -Description "Hierarchical global+local without stable day summary branch in router/pooling." -Args (
      $CommonHierArgs + @(
        "--router_summary_source", "none",
        "--pooling_summary_source", "none",
        "--label", "global_local",
        "--router_use_global_state", "1",
        "--router_use_local_state", "1",
        "--film_use_global_state", "1",
        "--film_use_local_state", "1",
        "--pooling_use_global_state", "1",
        "--pooling_use_local_state", "1"
      )
    )),
  (New-Variant -Order 4 -Label "global_only" -Description "Hierarchical ablation with only explicit global conditioning in FiLM/router/pooling." -Args (
      $CommonHierArgs + @(
        "--router_summary_source", "none",
        "--pooling_summary_source", "none",
        "--label", "global_only",
        "--router_use_global_state", "1",
        "--router_use_local_state", "0",
        "--film_use_global_state", "1",
        "--film_use_local_state", "0",
        "--pooling_use_global_state", "1",
        "--pooling_use_local_state", "0"
      )
    )),
  (New-Variant -Order 5 -Label "local_only_residual" -Description "Hierarchical ablation with only explicit local conditioning in FiLM/router/pooling." -Args (
      $CommonHierArgs + @(
        "--router_summary_source", "none",
        "--pooling_summary_source", "none",
        "--label", "local_only_residual",
        "--router_use_global_state", "0",
        "--router_use_local_state", "1",
        "--film_use_global_state", "0",
        "--film_use_local_state", "1",
        "--pooling_use_global_state", "0",
        "--pooling_use_local_state", "1"
      )
    ))
)

Write-Host "=== Hierarchical State Field Ablation Plan ==="
Write-Host "RepoRoot: $RepoRoot"
Write-Host "PythonExe: $PythonExe"
Write-Host "MarketStatePath: $MarketStatePath"
Write-Host "MarketDaySummaryPath: $MarketDaySummaryPath"
Write-Host "Seed: $Seed"
Write-Host "RunTag: $RunTag"
Write-Host "Execute: $Execute"
Write-Host ""
Write-Host "Execution order:"
foreach ($Variant in $Variants) {
  Write-Host ("  {0}. {1} - {2}" -f $Variant.Order, $Variant.Label, $Variant.Description)
}
Write-Host ""

$VariantCommands = @()
foreach ($Variant in $Variants) {
  $ScriptArgs = @((Join-Path $RepoRoot "scripts/run_workflow_market_state_variant.py")) + $Variant.Args
  $VariantCommands += [ordered]@{
    label = $Variant.Label
    description = $Variant.Description
    command = (Format-Command -Exe $PythonExe -Args $ScriptArgs)
  }
}

$CompareMd = Join-Path $CompareOutRoot "csi300_hierarchical_state_field_comparison.md"
$CompareCsv = Join-Path $CompareOutRoot "csi300_hierarchical_state_field_comparison.csv"

if (-not $Execute) {
  Write-Host "Dry-run mode. Commands are printed only; nothing will execute."
  Write-Host ""
  foreach ($Entry in $VariantCommands) {
    Write-Host ("[{0}]" -f $Entry.label)
    Write-Host $Entry.command
    Write-Host ""
  }
  Write-Host "[compare template]"
  Write-Host (Format-Command -Exe $PythonExe -Args @(
      (Join-Path $RepoRoot "scripts/compare_market_state_runs.py"),
      "--base", "<RUN_ID_BASE>",
      "--global_only", "<RUN_ID_GLOBAL_ONLY>",
      "--local_only_residual", "<RUN_ID_LOCAL_ONLY_RESIDUAL>",
      "--global_local", "<RUN_ID_GLOBAL_LOCAL>",
      "--global_local_stable_summary", "<RUN_ID_GLOBAL_LOCAL_STABLE_SUMMARY>",
      "--out_md", $CompareMd,
      "--out_csv", $CompareCsv
    ))
  exit 0
}

function Invoke-Variant {
  param(
    [Parameter(Mandatory = $true)][hashtable]$Variant
  )

  $LogPath = Join-Path $RunLogRoot ("{0:00}_{1}.log" -f $Variant.Order, $Variant.Label)
  $Args = @((Join-Path $RepoRoot "scripts/run_workflow_market_state_variant.py")) + $Variant.Args

  Write-Host ("=== Running {0}: {1} ===" -f $Variant.Order, $Variant.Label)
  Write-Host (Format-Command -Exe $PythonExe -Args $Args)

  $Lines = & $PythonExe @Args 2>&1 | Tee-Object -FilePath $LogPath
  if ($LASTEXITCODE -ne 0) {
    throw "Variant failed: $($Variant.Label). See log: $LogPath"
  }

  $RecorderLine = $Lines | Where-Object { $_ -match '>>> \[Variant\] recorder_id=(.+)$' } | Select-Object -Last 1
  if ($null -eq $RecorderLine) {
    throw "Unable to parse recorder_id for $($Variant.Label). See log: $LogPath"
  }

  $RecorderId = ([string]$RecorderLine) -replace '^.*recorder_id=', ''

  return [ordered]@{
    label = $Variant.Label
    recorder_id = $RecorderId.Trim()
    log_path = $LogPath
  }
}

$Results = [ordered]@{}
foreach ($Variant in $Variants) {
  $Result = Invoke-Variant -Variant $Variant
  $Results[$Variant.Label] = $Result
}

$Manifest = [ordered]@{
  run_tag = $RunTag
  python = $PythonExe
  market_state_path = $MarketStatePath
  market_day_summary_path = $MarketDaySummaryPath
  seed = $Seed
  variants = @($Results.Values)
  compare_out_md = $CompareMd
  compare_out_csv = $CompareCsv
}

$ManifestPath = Join-Path $RunLogRoot "run_manifest.json"
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding UTF8
Write-Host "Wrote run manifest: $ManifestPath"

if ($SkipCompare) {
  Write-Host "SkipCompare set. Compare step was not executed."
  exit 0
}

$CompareArgs = @(
  (Join-Path $RepoRoot "scripts/compare_market_state_runs.py"),
  "--base", $Results["base"].recorder_id,
  "--global_only", $Results["global_only"].recorder_id,
  "--local_only_residual", $Results["local_only_residual"].recorder_id,
  "--global_local", $Results["global_local"].recorder_id,
  "--global_local_stable_summary", $Results["global_local_stable_summary"].recorder_id,
  "--out_md", $CompareMd,
  "--out_csv", $CompareCsv
)

Write-Host "=== Running compare ==="
Write-Host (Format-Command -Exe $PythonExe -Args $CompareArgs)
& $PythonExe @CompareArgs
if ($LASTEXITCODE -ne 0) {
  throw "Compare step failed."
}

Write-Host "All variants completed."
Write-Host "Comparison markdown: $CompareMd"
Write-Host "Comparison csv: $CompareCsv"
