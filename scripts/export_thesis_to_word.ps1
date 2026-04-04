param(
  [Parameter(Mandatory = $true)]
  [string]$MdPath,

  [Parameter(Mandatory = $true)]
  [string]$DocxPath,

  [Parameter(Mandatory = $false)]
  [string]$HtmlPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function To-HtmlEncodedInline {
  param([Parameter(Mandatory = $true)][string]$Text)
  $s = [System.Net.WebUtility]::HtmlEncode($Text)
  # bold
  $s = [System.Text.RegularExpressions.Regex]::Replace($s, '\*\*(.+?)\*\*', '<strong>$1</strong>')
  # inline code
  $s = [System.Text.RegularExpressions.Regex]::Replace($s, '`([^`]+)`', '<code>$1</code>')
  return $s
}

function Normalize-PathForJoin {
  param([Parameter(Mandatory = $true)][string]$Path)
  return ($Path -replace "/", "\\")
}

function To-FileUri {
  param([Parameter(Mandatory = $true)][string]$Path)
  $full = [IO.Path]::GetFullPath($Path)
  return (New-Object System.Uri($full)).AbsoluteUri
}

function Parse-MarkdownTable {
  param(
    [string[]]$Lines,
    [int]$StartIndex
  )

  $headerLine = $Lines[$StartIndex]
  $sepLine = $Lines[$StartIndex + 1]
  $rows = New-Object System.Collections.Generic.List[object]

  $split = {
    param([string]$line)
    $trimmed = $line.Trim()
    if ($trimmed.StartsWith("|")) { $trimmed = $trimmed.Substring(1) }
    if ($trimmed.EndsWith("|")) { $trimmed = $trimmed.Substring(0, $trimmed.Length - 1) }
    return ($trimmed -split '\|') | ForEach-Object { $_.Trim() }
  }

  $header = & $split $headerLine
  $index = $StartIndex + 2
  while ($index -lt $Lines.Length) {
    $line = $Lines[$index]
    if ([string]::IsNullOrWhiteSpace($line)) { break }
    if ($line -notmatch '\|') { break }
    $rows.Add((& $split $line))
    $index++
  }

  return [pscustomobject]@{
    Header = $header
    Rows = $rows
    NextIndex = $index
  }
}

if (-not (Test-Path $MdPath)) {
  throw "Markdown not found: $MdPath"
}

$mdFull = [IO.Path]::GetFullPath($MdPath)
$mdDir = [IO.Path]::GetDirectoryName($mdFull)

if ([string]::IsNullOrWhiteSpace($HtmlPath)) {
  $HtmlPath = [IO.Path]::ChangeExtension($DocxPath, ".html")
}

$lines = Get-Content -Encoding UTF8 $mdFull

$html = New-Object System.Collections.Generic.List[string]
$html.Add("<!doctype html>")
$html.Add("<html><head><meta charset=""utf-8"">")
$html.Add("<style>")
$html.Add("body { font-family: 'Times New Roman', 'Microsoft YaHei', serif; line-height: 1.5; }")
$html.Add("h1, h2, h3, h4 { font-family: 'Microsoft YaHei', 'Times New Roman', serif; }")
$html.Add("table { border-collapse: collapse; width: 100%; margin: 8px 0; }")
$html.Add("th, td { border: 1px solid #888; padding: 6px 8px; vertical-align: top; }")
$html.Add("code { font-family: Consolas, monospace; background: #f3f3f3; padding: 1px 3px; }")
$html.Add("pre { font-family: Consolas, monospace; background: #f3f3f3; padding: 10px; white-space: pre-wrap; }")
$html.Add("img { max-width: 100%; height: auto; }")
$html.Add("</style></head><body>")

$inCode = $false
$codeLang = ""
$codeBuffer = New-Object System.Collections.Generic.List[string]

$inUl = $false
$paraBuffer = New-Object System.Collections.Generic.List[string]

function Flush-Paragraph {
  param([System.Collections.Generic.List[string]]$buf, [System.Collections.Generic.List[string]]$out)
  if ($buf.Count -eq 0) { return }
  $joined = ($buf | ForEach-Object { To-HtmlEncodedInline $_ }) -join "<br/>"
  $out.Add("<p>$joined</p>")
  $buf.Clear()
}

function Close-Ul {
  param([ref]$flag, [System.Collections.Generic.List[string]]$out)
  if ($flag.Value) {
    $out.Add("</ul>")
    $flag.Value = $false
  }
}

$i = 0
while ($i -lt $lines.Length) {
  $line = $lines[$i]

  if ($line.StartsWith('```')) {
    if ($inCode) {
      $codeText = [System.Net.WebUtility]::HtmlEncode(($codeBuffer -join [Environment]::NewLine))
      if (-not ($codeLang -and $codeLang.ToLowerInvariant() -eq "mermaid")) {
        $html.Add("<pre><code>$codeText</code></pre>")
      }
      $codeBuffer.Clear()
      $inCode = $false
      $codeLang = ""
      $i++
      continue
    }

    Flush-Paragraph $paraBuffer $html
    Close-Ul ([ref]$inUl) $html

    $inCode = $true
    $codeLang = $line.Substring(3).Trim()
    $i++
    continue
  }

  if ($inCode) {
    $codeBuffer.Add($line)
    $i++
    continue
  }

  if ([string]::IsNullOrWhiteSpace($line)) {
    Flush-Paragraph $paraBuffer $html
    Close-Ul ([ref]$inUl) $html
    $i++
    continue
  }

  if ($line.Trim() -eq "---") {
    Flush-Paragraph $paraBuffer $html
    Close-Ul ([ref]$inUl) $html
    $html.Add("<hr/>")
    $i++
    continue
  }

  if ($line -match '^(#{1,6})\s+(.*)$') {
    Flush-Paragraph $paraBuffer $html
    Close-Ul ([ref]$inUl) $html
    $level = $Matches[1].Length
    $content = To-HtmlEncodedInline $Matches[2]
    $html.Add("<h$level>$content</h$level>")
    $i++
    continue
  }

  # Markdown table
  if (($line -match '\|') -and ($i + 1 -lt $lines.Length) -and ($lines[$i + 1] -match '^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$')) {
    Flush-Paragraph $paraBuffer $html
    Close-Ul ([ref]$inUl) $html

    $tbl = Parse-MarkdownTable -Lines $lines -StartIndex $i
    $html.Add("<table><thead><tr>")
    foreach ($cell in $tbl.Header) {
      $html.Add("<th>$(To-HtmlEncodedInline $cell)</th>")
    }
    $html.Add("</tr></thead><tbody>")
    foreach ($row in $tbl.Rows) {
      $html.Add("<tr>")
      foreach ($cell in $row) {
        $html.Add("<td>$(To-HtmlEncodedInline $cell)</td>")
      }
      $html.Add("</tr>")
    }
    $html.Add("</tbody></table>")
    $i = $tbl.NextIndex
    continue
  }

  # Image
  if ($line -match '!\[(.*?)\]\((.*?)\)') {
    Flush-Paragraph $paraBuffer $html
    Close-Ul ([ref]$inUl) $html
    $alt = $Matches[1]
    $srcRaw = $Matches[2]
    $src = $srcRaw
    if ($src -notmatch "^(https?://|file:/)") {
      $abs = [IO.Path]::GetFullPath((Join-Path $mdDir (Normalize-PathForJoin $src)))
      $src = (New-Object System.Uri($abs)).AbsoluteUri
    }
    $html.Add("<p><img src=""$src"" alt=""$([System.Net.WebUtility]::HtmlEncode($alt))""></p>")
    $i++
    continue
  }

  # Unordered list
  if ($line -match '^\s*[-*]\s+(.+)$') {
    Flush-Paragraph $paraBuffer $html
    if (-not $inUl) {
      $html.Add("<ul>")
      $inUl = $true
    }
    $item = To-HtmlEncodedInline $Matches[1]
    $html.Add("<li>$item</li>")
    $i++
    continue
  }

  # Default: paragraph content
  $paraBuffer.Add($line.TrimEnd())
  $i++
}

Flush-Paragraph $paraBuffer $html
Close-Ul ([ref]$inUl) $html

$html.Add("</body></html>")

$htmlFull = [IO.Path]::GetFullPath($HtmlPath)
$docxFull = [IO.Path]::GetFullPath($DocxPath)

$outHtmlDir = [IO.Path]::GetDirectoryName($htmlFull)
if ($outHtmlDir -and -not (Test-Path $outHtmlDir)) { New-Item -ItemType Directory -Force -Path $outHtmlDir | Out-Null }
$outDocxDir = [IO.Path]::GetDirectoryName($docxFull)
if ($outDocxDir -and -not (Test-Path $outDocxDir)) { New-Item -ItemType Directory -Force -Path $outDocxDir | Out-Null }

Set-Content -Encoding UTF8 -Path $htmlFull -Value ($html -join "`n")

# Export to Word via COM
$word = $null
$doc = $null
try {
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0

  $doc = $word.Documents.Open($htmlFull)
  # wdFormatXMLDocument = 12
  $doc.SaveAs2($docxFull, 12)
} finally {
  if ($doc -ne $null) {
    try { $doc.Close($false) | Out-Null } catch { }
    try { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) } catch { }
  }
  if ($word -ne $null) {
    try { $word.Quit() | Out-Null } catch { }
    try { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) } catch { }
  }
  try { [GC]::Collect(); [GC]::WaitForPendingFinalizers() } catch { }
}

Write-Host "Wrote HTML: $htmlFull"
Write-Host "Wrote DOCX: $docxFull"
