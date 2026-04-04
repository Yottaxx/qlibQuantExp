param(
  [Parameter(Mandatory = $false)]
  [string]$HtmlPath = "analysis/thesis_outputs/evidence_e395e6d51e8e4c88b58de249342c085f/html/qlib_analysis_position_report_graph.html",

  [Parameter(Mandatory = $false)]
  [string]$OutputPath = "analysis/thesis_outputs/evidence_e395e6d51e8e4c88b58de249342c085f/figures/position_cumret_vs_bench_test.png",

  [Parameter(Mandatory = $false)]
  [int]$Width = 1600,

  [Parameter(Mandatory = $false)]
  [int]$Height = 900,

  [Parameter(Mandatory = $false)]
  [switch]$WithCost = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-PlotlyDataJsonArray {
  param([Parameter(Mandatory = $true)][string]$Text)

  $idx = $Text.IndexOf("Plotly.newPlot")
  if ($idx -lt 0) {
    throw "Cannot find Plotly.newPlot(...) in HTML."
  }

  $start = $Text.IndexOf("[", $idx)
  if ($start -lt 0) {
    throw "Cannot find start of data array ([) after Plotly.newPlot."
  }

  $depth = 0
  $inString = $false
  $escape = $false
  $end = -1

  for ($i = $start; $i -lt $Text.Length; $i++) {
    $ch = $Text[$i]

    if ($escape) {
      $escape = $false
      continue
    }

    if ($inString) {
      if ($ch -eq "\") {
        $escape = $true
        continue
      }
      if ($ch -eq '"') {
        $inString = $false
      }
      continue
    }

    if ($ch -eq '"') {
      $inString = $true
      continue
    }

    if ($ch -eq "[") {
      $depth++
      continue
    }

    if ($ch -eq "]") {
      $depth--
      if ($depth -eq 0) {
        $end = $i
        break
      }
    }
  }

  if ($end -lt 0) {
    throw "Cannot find end of data array (]) after Plotly.newPlot."
  }

  return $Text.Substring($start, $end - $start + 1)
}

function Convert-PlotlyTypedArrayToDoubleArray {
  param([Parameter(Mandatory = $true)]$Y)

  if ($Y -is [System.Array]) {
    return [double[]]($Y | ForEach-Object { [double]$_ })
  }

  $dtype = $Y.dtype
  $bdata = $Y.bdata
  if ([string]::IsNullOrWhiteSpace($dtype) -or [string]::IsNullOrWhiteSpace($bdata)) {
    throw "Unexpected Plotly typed-array format (missing dtype/bdata)."
  }

  $bytes = [Convert]::FromBase64String($bdata)
  switch ($dtype) {
    "f4" {
      $elemSize = 4
      $toNumber = { param([byte[]]$b, [int]$o) [double]([BitConverter]::ToSingle($b, $o)) }
    }
    "f8" {
      $elemSize = 8
      $toNumber = { param([byte[]]$b, [int]$o) [double]([BitConverter]::ToDouble($b, $o)) }
    }
    default {
      throw "Unsupported dtype: $dtype"
    }
  }

  if (($bytes.Length % $elemSize) -ne 0) {
    $bytes = $bytes[0..([Math]::Floor($bytes.Length / $elemSize) * $elemSize - 1)]
  }

  $n = [Math]::Floor($bytes.Length / $elemSize)
  $arr = New-Object double[] $n
  for ($i = 0; $i -lt $n; $i++) {
    $arr[$i] = & $toNumber $bytes ($i * $elemSize)
  }
  return $arr
}

function Convert-PlotlyDateArrayToDateTimeArray {
  param([Parameter(Mandatory = $true)][object[]]$X)

  $dates = New-Object "System.DateTime[]" $X.Length
  $firstDate = $null
  $t0Idx = @()

  for ($i = 0; $i -lt $X.Length; $i++) {
    $s = [string]$X[$i]
    if ($s -match '^\d{4}-\d{2}-\d{2}') {
      $dt = [DateTime]::ParseExact($s.Substring(0, 10), "yyyy-MM-dd", [System.Globalization.CultureInfo]::InvariantCulture)
      $dates[$i] = $dt
      if ($null -eq $firstDate) { $firstDate = $dt }
      continue
    }

    if ($s -eq 'T0') {
      $t0Idx += $i
      continue
    }

    throw "Unexpected x[$i] value: $s"
  }

  if ($t0Idx.Count -gt 0) {
    if ($null -eq $firstDate) {
      throw "Cannot resolve T0 without any valid dates in x."
    }
    foreach ($i in $t0Idx) {
      $dates[$i] = $firstDate.AddDays(-1)
    }
  }

  return $dates
}

if (-not (Test-Path $HtmlPath)) {
  throw "HTML not found: $HtmlPath"
}

$html = Get-Content -Raw -Encoding UTF8 $HtmlPath
$dataJson = Get-PlotlyDataJsonArray -Text $html
$data = $dataJson | ConvertFrom-Json

$suffix = if ($WithCost) { "w cost" } else { "wo cost" }

$traceNames = @(
  "cum return $suffix",
  "cum bench",
  "cum ex return $suffix"
)

$nameMap = @{
  "cum return $suffix"    = if ($WithCost) { "Strategy Cumulative Return (with cost)" } else { "Strategy Cumulative Return (no cost)" }
  "cum bench"             = "Benchmark Cumulative Return"
  "cum ex return $suffix" = if ($WithCost) { "Excess Cumulative Return (with cost)" } else { "Excess Cumulative Return (no cost)" }
}

$selected = @()
foreach ($name in $traceNames) {
  $tr = $data | Where-Object { $_.name -eq $name } | Select-Object -First 1
  if ($tr) { $selected += $tr }
}

if (-not $selected -or $selected.Count -lt 2) {
  throw "Cannot find required traces in position report plot. Found: $($data | ForEach-Object {$_.name} | Sort-Object | Get-Unique -join ', ')"
}

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms.DataVisualization

$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width = $Width
$chart.Height = $Height
$chart.BackColor = [System.Drawing.Color]::White

$chartArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea "Main"
$chartArea.AxisX.LabelStyle.Format = "yyyy-MM"
$chartArea.AxisX.LabelStyle.Angle = -45
$chartArea.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
$chartArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
$chartArea.AxisX.Interval = 3
$chartArea.AxisX.IntervalType = [System.Windows.Forms.DataVisualization.Charting.DateTimeIntervalType]::Months
$chartArea.AxisX.Title = "Date"
$chartArea.AxisY.Title = "Cumulative Return"
$chart.ChartAreas.Add($chartArea)

$legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
$legend.Alignment = [System.Drawing.StringAlignment]::Center
$chart.Legends.Add($legend)

$style = @{
  "cum bench" = @{ Color = [System.Drawing.Color]::FromArgb(120, 120, 120); Dash = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Dash; Width = 2 }
}

$palette = @(
  [System.Drawing.Color]::FromArgb(31, 119, 180),
  [System.Drawing.Color]::FromArgb(214, 39, 40),
  [System.Drawing.Color]::FromArgb(44, 160, 44)
)

for ($idx = 0; $idx -lt $selected.Count; $idx++) {
  $trace = $selected[$idx]
  $xDates = Convert-PlotlyDateArrayToDateTimeArray -X $trace.x
  $yVals = Convert-PlotlyTypedArrayToDoubleArray -Y $trace.y
  $n = [Math]::Min($xDates.Length, $yVals.Length)

  $seriesName = if ($nameMap.ContainsKey($trace.name)) { $nameMap[$trace.name] } else { $trace.name }
  $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series $seriesName
  $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
  $series.BorderWidth = 3
  $series.XValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::DateTime
  $series.YValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::Double
  $series.Color = $palette[[Math]::Min($idx, $palette.Count - 1)]

  if ($style.ContainsKey($trace.name)) {
    $series.Color = $style[$trace.name].Color
    $series.BorderDashStyle = $style[$trace.name].Dash
    $series.BorderWidth = $style[$trace.name].Width
  }

  if ($trace.name -like "cum ex return*") {
    $series.BorderDashStyle = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Dot
    $series.BorderWidth = 2
  }

  for ($i = 0; $i -lt $n; $i++) {
    [void]$series.Points.AddXY($xDates[$i], $yVals[$i])
  }

  $chart.Series.Add($series)
}

$title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$title.Text = if ($WithCost) { "Strategy vs Benchmark (With Cost)" } else { "Strategy vs Benchmark (No Cost)" }
$title.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
$chart.Titles.Add($title)

$outDir = Split-Path -Parent $OutputPath
if ($outDir -and -not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$chart.SaveImage($OutputPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
Write-Host "Wrote PNG: $OutputPath"
