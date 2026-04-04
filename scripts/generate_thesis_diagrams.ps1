param(
  [Parameter(Mandatory = $false)]
  [string]$OutDir = "analysis/figures"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

function New-Color($hex) {
  return [System.Drawing.ColorTranslator]::FromHtml($hex)
}

function New-Font([string]$name, [float]$size, [System.Drawing.FontStyle]$style = [System.Drawing.FontStyle]::Regular) {
  return New-Object System.Drawing.Font($name, $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
}

function Draw-Box {
  param(
    [Parameter(Mandatory = $true)][System.Drawing.Graphics]$G,
    [Parameter(Mandatory = $true)][System.Drawing.RectangleF]$Rect,
    [Parameter(Mandatory = $true)][string]$Text,
    [Parameter(Mandatory = $true)][System.Drawing.Brush]$Fill,
    [Parameter(Mandatory = $true)][System.Drawing.Pen]$Border,
    [Parameter(Mandatory = $true)][System.Drawing.Font]$Font
  )

  $G.FillRectangle($Fill, $Rect)
  $G.DrawRectangle($Border, $Rect.X, $Rect.Y, $Rect.Width, $Rect.Height)

  $sf = New-Object System.Drawing.StringFormat
  $sf.Alignment = [System.Drawing.StringAlignment]::Center
  $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
  $sf.Trimming = [System.Drawing.StringTrimming]::EllipsisWord

  $G.DrawString($Text, $Font, [System.Drawing.Brushes]::Black, $Rect, $sf)
}

function Draw-Arrow {
  param(
    [Parameter(Mandatory = $true)][System.Drawing.Graphics]$G,
    [Parameter(Mandatory = $true)][System.Drawing.PointF]$From,
    [Parameter(Mandatory = $true)][System.Drawing.PointF]$To,
    [Parameter(Mandatory = $false)][System.Drawing.Color]$Color = [System.Drawing.Color]::FromArgb(80, 80, 80),
    [Parameter(Mandatory = $false)][float]$Width = 3.0
  )

  $pen = New-Object System.Drawing.Pen($Color, $Width)
  $pen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
  $pen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
  $arrow = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(6, 8, $true)
  $pen.CustomEndCap = $arrow
  $G.DrawLine($pen, $From, $To)
  $pen.Dispose()
}

function Ensure-Dir([string]$path) {
  if (-not (Test-Path $path)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
  }
}

Ensure-Dir $OutDir

function Draw-Fig3_1([string]$outPath) {
  $w = 1800
  $h = 1000
  $bmp = New-Object System.Drawing.Bitmap -ArgumentList $w, $h
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
  $g.Clear([System.Drawing.Color]::White)

  $titleFont = New-Font "Microsoft YaHei" 26 ([System.Drawing.FontStyle]::Bold)
  $boxFont = New-Font "Microsoft YaHei" 18
  $smallFont = New-Font "Microsoft YaHei" 14

  $g.DrawString("Figure 3-1  Method Framework (Slow sets direction; Fast adjusts pace)", $titleFont, [System.Drawing.Brushes]::Black, 60, 30)
  $g.DrawString("Expert-led direction + model-assisted signals -> three lists -> executable closed loop", $smallFont, [System.Drawing.Brushes]::DimGray, 60, 70)

  $border = New-Object System.Drawing.Pen((New-Color "#2F5597"), 3)
  $border2 = New-Object System.Drawing.Pen((New-Color "#1F7A5C"), 3)
  $border3 = New-Object System.Drawing.Pen((New-Color "#7A4E00"), 3)
  $border4 = New-Object System.Drawing.Pen((New-Color "#5A5A5A"), 3)

  $slowRect = New-Object System.Drawing.RectangleF -ArgumentList 80, 170, 420, 210
  $dataRect = New-Object System.Drawing.RectangleF -ArgumentList 80, 440, 420, 260
  $fastRect = New-Object System.Drawing.RectangleF -ArgumentList 560, 320, 520, 250
  $listRect = New-Object System.Drawing.RectangleF -ArgumentList 1160, 250, 420, 380
  $actRect = New-Object System.Drawing.RectangleF -ArgumentList 1600, 250, 160, 380

  Draw-Box -G $g -Rect $slowRect -Text "Slow Variables (Expert-led)`nNational strategy / industrial policy / annual focus`nAdmission rules / limits / risk floor" -Fill ([System.Drawing.Brushes]::AliceBlue) -Border $border -Font $boxFont
  Draw-Box -G $g -Rect $dataRect -Text "Data Base (Governed)`nExternal: market / industry / macro`nInternal: client / project / credit / post-loan`nStandards, quality control, access control" -Fill ([System.Drawing.Brushes]::Honeydew) -Border $border2 -Font $boxFont
  Draw-Box -G $g -Rect $fastRect -Text "Fast Variables (Model-assisted)`nMarket state (vol / tail / crowding)`nAdaptive ranking (relative priority)`nOutput: percentile score (0~1)" -Fill ([System.Drawing.Brushes]::Moccasin) -Border $border3 -Font $boxFont
  Draw-Box -G $g -Rect $listRect -Text "Three Lists (Standardized Outputs)`n1) Industry rhythm (pace & priority)`n2) Risk warning (persistent weakness)`n3) Crowding warning (over-consensus)" -Fill ([System.Drawing.Brushes]::WhiteSmoke) -Border $border4 -Font $boxFont
  Draw-Box -G $g -Rect $actRect -Text "Actions`n& Review" -Fill ([System.Drawing.Brushes]::WhiteSmoke) -Border $border4 -Font $boxFont

  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $slowRect.Right, ($slowRect.Top + $slowRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $fastRect.Left, ($fastRect.Top + 60))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $dataRect.Right, ($dataRect.Top + $dataRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $fastRect.Left, ($fastRect.Top + 190))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $fastRect.Right, ($fastRect.Top + $fastRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $listRect.Left, ($listRect.Top + $listRect.Height/2))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $listRect.Right, ($listRect.Top + $listRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $actRect.Left, ($actRect.Top + $actRect.Height/2))

  # feedback arrow
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList ($actRect.Left + 20), ($actRect.Bottom - 40)) -To (New-Object System.Drawing.PointF -ArgumentList ($dataRect.Left + 40), ($dataRect.Top - 10)) -Color ([System.Drawing.Color]::FromArgb(120, 120, 120)) -Width 2.5
  $g.DrawString("Feedback: effect -> deviation -> iteration", $smallFont, [System.Drawing.Brushes]::DimGray, 420, 760)

  $bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose()
  $bmp.Dispose()
}

function Draw-Fig3_2([string]$outPath) {
  $w = 1800
  $h = 760
  $bmp = New-Object System.Drawing.Bitmap -ArgumentList $w, $h
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
  $g.Clear([System.Drawing.Color]::White)

  $titleFont = New-Font "Microsoft YaHei" 26 ([System.Drawing.FontStyle]::Bold)
  $boxFont = New-Font "Microsoft YaHei" 18
  $smallFont = New-Font "Microsoft YaHei" 14

  $g.DrawString("Figure 3-2  RST-MoE Logic (state-driven two-channel fusion)", $titleFont, [System.Drawing.Brushes]::Black, 60, 30)
  $g.DrawString("Management view: dynamically balance time-series vs cross-section signals under market states.", $smallFont, [System.Drawing.Brushes]::DimGray, 60, 70)

  $border = New-Object System.Drawing.Pen((New-Color "#2F5597"), 3)
  $border2 = New-Object System.Drawing.Pen((New-Color "#7A4E00"), 3)
  $border3 = New-Object System.Drawing.Pen((New-Color "#5A5A5A"), 3)

  $msRect = New-Object System.Drawing.RectangleF -ArgumentList 80, 160, 360, 140
  $xRect  = New-Object System.Drawing.RectangleF -ArgumentList 80, 360, 360, 140
  $routerRect = New-Object System.Drawing.RectangleF -ArgumentList 480, 250, 380, 180
  $timeRect = New-Object System.Drawing.RectangleF -ArgumentList 920, 160, 360, 140
  $crossRect = New-Object System.Drawing.RectangleF -ArgumentList 920, 360, 360, 140
  $fusionRect = New-Object System.Drawing.RectangleF -ArgumentList 1320, 250, 300, 180
  $scoreRect = New-Object System.Drawing.RectangleF -ArgumentList 1660, 250, 110, 180

  Draw-Box -G $g -Rect $msRect -Text "Market State (context)`nVol / Tail / Crowding / Theme" -Fill ([System.Drawing.Brushes]::Moccasin) -Border $border2 -Font $boxFont
  Draw-Box -G $g -Rect $xRect -Text "Input: historical feature window`n(factors / price-volume / structure)" -Fill ([System.Drawing.Brushes]::AliceBlue) -Border $border -Font $boxFont
  Draw-Box -G $g -Rect $routerRect -Text "State Router (interpretable)`nweight: w (time vs cross)`nentropy: H (over-extreme?)`nmemory: tau / half-life" -Fill ([System.Drawing.Brushes]::WhiteSmoke) -Border $border3 -Font $boxFont
  Draw-Box -G $g -Rect $timeRect -Text "Time channel`n(trend / persistence / lag)" -Fill ([System.Drawing.Brushes]::AliceBlue) -Border $border -Font $boxFont
  Draw-Box -G $g -Rect $crossRect -Text "Cross channel`n(relative strength / repricing)" -Fill ([System.Drawing.Brushes]::AliceBlue) -Border $border -Font $boxFont
  Draw-Box -G $g -Rect $fusionRect -Text "Fusion score`n s = w·s_time + (1-w)·s_cross " -Fill ([System.Drawing.Brushes]::WhiteSmoke) -Border $border3 -Font $boxFont
  Draw-Box -G $g -Rect $scoreRect -Text "Output`npercentile`nthree lists" -Fill ([System.Drawing.Brushes]::WhiteSmoke) -Border $border3 -Font $boxFont

  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $msRect.Right, ($msRect.Top + $msRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $routerRect.Left, ($routerRect.Top + 50))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $xRect.Right, ($xRect.Top + 30)) -To (New-Object System.Drawing.PointF -ArgumentList $timeRect.Left, ($timeRect.Top + 70))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $xRect.Right, ($xRect.Bottom - 30)) -To (New-Object System.Drawing.PointF -ArgumentList $crossRect.Left, ($crossRect.Top + 70))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $routerRect.Right, ($routerRect.Top + $routerRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $fusionRect.Left, ($fusionRect.Top + $fusionRect.Height/2))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $timeRect.Right, ($timeRect.Top + $timeRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $fusionRect.Left, ($fusionRect.Top + 50))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $crossRect.Right, ($crossRect.Top + $crossRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $fusionRect.Left, ($fusionRect.Bottom - 50))
  Draw-Arrow -G $g -From (New-Object System.Drawing.PointF -ArgumentList $fusionRect.Right, ($fusionRect.Top + $fusionRect.Height/2)) -To (New-Object System.Drawing.PointF -ArgumentList $scoreRect.Left, ($scoreRect.Top + $scoreRect.Height/2))

  $bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose()
  $bmp.Dispose()
}

Draw-Fig3_1 (Join-Path $OutDir "fig3_1_method_architecture.png")
Draw-Fig3_2 (Join-Path $OutDir "fig3_2_rst_moe_logic.png")

Write-Host "Wrote diagrams to: $OutDir"
