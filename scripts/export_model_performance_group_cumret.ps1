param(
  [Parameter(Mandatory = $false)]
  [string]$HtmlPath = "analysis/thesis_outputs/evidence_e395e6d51e8e4c88b58de249342c085f/html/qlib_analysis_model_model_performance_graph_1.html",

  [Parameter(Mandatory = $false)]
  [string]$OutputPath = "analysis/thesis_outputs/evidence_e395e6d51e8e4c88b58de249342c085f/figures/model_performance_group_cumret_test.png",

  [Parameter(Mandatory = $false)]
  [int]$Width = 1600,

  [Parameter(Mandatory = $false)]
  [int]$Height = 900,

  [Parameter(Mandatory = $false)]
  [switch]$IncludeLongShort = $true,

  [Parameter(Mandatory = $false)]
  [switch]$IncludeLongAverage = $true
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
  param([Parameter(Mandatory = $true)][string[]]$X)

  $dates = New-Object "System.DateTime[]" $X.Length
  for ($i = 0; $i -lt $X.Length; $i++) {
    $s = $X[$i]
    if ($null -eq $s -or $s.Length -lt 10) {
      throw "Unexpected date string in x[$i]: $s"
    }
    $dates[$i] = [DateTime]::ParseExact($s.Substring(0, 10), "yyyy-MM-dd", [System.Globalization.CultureInfo]::InvariantCulture)
  }
  return $dates
}

if (-not (Test-Path $HtmlPath)) {
  throw "HTML not found: $HtmlPath"
}

$html = Get-Content -Raw -Encoding UTF8 $HtmlPath
$dataJson = Get-PlotlyDataJsonArray -Text $html
$data = $dataJson | ConvertFrom-Json

$groupTraces =
  $data |
  Where-Object { $_.name -match "^Group[1-5]$" } |
  Sort-Object { [int]$_.name.Substring(5) }

if (-not $groupTraces -or $groupTraces.Count -lt 2) {
  throw "Cannot find Group1..Group5 traces in plotly data."
}

$extraTraces = @()
if ($IncludeLongShort) {
  $ls = $data | Where-Object { $_.name -eq "long-short" } | Select-Object -First 1
  if ($ls) { $extraTraces += $ls }
}
if ($IncludeLongAverage) {
  $la = $data | Where-Object { $_.name -eq "long-average" } | Select-Object -First 1
  if ($la) { $extraTraces += $la }
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

$palette = @(
  [System.Drawing.Color]::FromArgb(31, 119, 180),
  [System.Drawing.Color]::FromArgb(255, 127, 14),
  [System.Drawing.Color]::FromArgb(44, 160, 44),
  [System.Drawing.Color]::FromArgb(214, 39, 40),
  [System.Drawing.Color]::FromArgb(148, 103, 189)
)

$plotTraces = @()
$plotTraces += $groupTraces
$plotTraces += $extraTraces

for ($idx = 0; $idx -lt $plotTraces.Count; $idx++) {
  $trace = $plotTraces[$idx]
  $xDates = Convert-PlotlyDateArrayToDateTimeArray -X $trace.x
  $yVals = Convert-PlotlyTypedArrayToDoubleArray -Y $trace.y
  $n = [Math]::Min($xDates.Length, $yVals.Length)

  $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series $trace.name
  $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
  $series.BorderWidth = 2
  $series.XValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::DateTime
  $series.YValueType = [System.Windows.Forms.DataVisualization.Charting.ChartValueType]::Double

  if ($trace.name -match "^Group[1-5]$") {
    $groupIdx = [int]$trace.name.Substring(5) - 1
    $series.Color = $palette[[Math]::Min($groupIdx, $palette.Count - 1)]
  } elseif ($trace.name -eq "long-short") {
    $series.Color = [System.Drawing.Color]::Black
    $series.BorderWidth = 3
  } elseif ($trace.name -eq "long-average") {
    $series.Color = [System.Drawing.Color]::FromArgb(80, 80, 80)
    $series.BorderWidth = 3
    $series.BorderDashStyle = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Dash
  } else {
    $series.Color = [System.Drawing.Color]::DimGray
  }

  for ($i = 0; $i -lt $n; $i++) {
    [void]$series.Points.AddXY($xDates[$i], $yVals[$i])
  }

  $chart.Series.Add($series)
}

$title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$title.Text = "Cumulative Return by Score Group (Group1..Group5) + Spreads"
$title.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
$chart.Titles.Add($title)

$outDir = Split-Path -Parent $OutputPath
if ($outDir -and -not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$chart.SaveImage($OutputPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
Write-Host "Wrote PNG: $OutputPath"
