param(
    [string]$InputCsv = "",
    [string]$OutputSvg = "",
    [string]$OutputPng = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $InputCsv) {
    $InputCsv = Join-Path $root "analysis_cache\all_gpu_metrics_timeseries\RTX4090\C01_all_35_metrics_1ms.csv"
}
$outDir = Join-Path $root "analysis_outputs\figures"
[System.IO.Directory]::CreateDirectory($outDir) | Out-Null
if (-not $OutputSvg) { $OutputSvg = Join-Path $outDir "yolo_c01_gpu_metric_periodicity.svg" }
if (-not $OutputPng) { $OutputPng = Join-Path $outDir "yolo_c01_gpu_metric_periodicity.png" }

$startMs = 3150
$periodMs = 45
$cycleCount = 10
$endMs = $startMs + $periodMs * $cycleCount
$metrics = @(
    @{ Name = "SMs Active [Throughput %]"; Label = "SMs Active"; Color = "#42a5ff" },
    @{ Name = "SM Issue [Throughput %]"; Label = "SM Issue"; Color = "#2fd0bb" },
    @{ Name = "Tensor Active [Throughput %]"; Label = "Tensor Active"; Color = "#ffb84d" },
    @{ Name = "DRAM Read Bandwidth [Throughput %]"; Label = "DRAM Read"; Color = "#b27cff" },
    @{ Name = "DRAM Write Bandwidth [Throughput %]"; Label = "DRAM Write"; Color = "#ff7180" }
)

$rows = Import-Csv -LiteralPath $InputCsv | Where-Object {
    [int]$_.time_ms -ge $startMs -and [int]$_.time_ms -le $endMs
}
if ($rows.Count -lt 400) { throw "Selected time window is incomplete." }

function Escape-Xml([string]$text) {
    return [System.Security.SecurityElement]::Escape($text)
}

function Smooth-Values($values, [int]$radius = 2) {
    $result = New-Object double[] $values.Count
    for ($i = 0; $i -lt $values.Count; $i++) {
        $lo = [Math]::Max(0, $i - $radius)
        $hi = [Math]::Min($values.Count - 1, $i + $radius)
        $sum = 0.0
        for ($j = $lo; $j -le $hi; $j++) { $sum += $values[$j] }
        $result[$i] = $sum / ($hi - $lo + 1)
    }
    return $result
}

$W = 1600
$H = 900
$plotX = 235.0
$plotY = 205.0
$plotW = 1270.0
$laneH = 98.0
$laneGap = 12.0
$plotH = $metrics.Count * $laneH + ($metrics.Count - 1) * $laneGap

$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine('<?xml version="1.0" encoding="UTF-8"?>')
[void]$sb.AppendLine("<svg xmlns=`"http://www.w3.org/2000/svg`" width=`"$W`" height=`"$H`" viewBox=`"0 0 $W $H`">")
[void]$sb.AppendLine('<rect width="1600" height="900" fill="#ffffff"/>')
[void]$sb.AppendLine('<style>text{font-family:"Microsoft JhengHei","Noto Sans TC",Arial,sans-serif}.title{font-size:38px;font-weight:700;fill:#172033}.subtitle{font-size:18px;fill:#5d6b7b}.kicker{font-size:15px;font-weight:700;fill:#087f73;letter-spacing:1.5px}.label{font-size:18px;font-weight:700;fill:#172033}.small{font-size:14px;fill:#667789}.axis{font-size:13px;fill:#667789}.callout{font-size:17px;font-weight:700;fill:#c96a05}</style>')
[void]$sb.AppendLine('<text x="74" y="62" class="kicker">REAL YOLO26 PROFILING TRACE</text>')
[void]$sb.AppendLine('<text x="74" y="116" class="title">YOLO26 C01 的 GPU 特徵呈現規律週期</text>')
[void]$sb.AppendLine('<text x="74" y="153" class="subtitle">RTX4090 · Nsight GPU Metrics · 1 ms sampling · marker-free detected period = 45 ms</text>')

# Alternating cycle bands and boundaries.
for ($c = 0; $c -lt $cycleCount; $c++) {
    $x = $plotX + $plotW * ($c / $cycleCount)
    $cw = $plotW / $cycleCount
    if ($c % 2 -eq 0) {
        [void]$sb.AppendLine(("<rect x=`"{0:F1}`" y=`"{1:F1}`" width=`"{2:F1}`" height=`"{3:F1}`" rx=`"8`" fill=`"#e7eff7`" opacity=`"0.82`"/>" -f $x,$plotY,$cw,$plotH))
    }
    [void]$sb.AppendLine(("<line x1=`"{0:F1}`" y1=`"{1:F1}`" x2=`"{0:F1}`" y2=`"{2:F1}`" stroke=`"#a9b8c7`" stroke-width=`"1`" stroke-dasharray=`"5 6`" opacity=`"0.75`"/>" -f $x,$plotY,($plotY+$plotH)))
    [void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"{1:F1}`" class=`"axis`" text-anchor=`"middle`">cycle {2}</text>" -f ($x+$cw/2),($plotY-12),($c+1)))
}
[void]$sb.AppendLine(("<line x1=`"{0:F1}`" y1=`"{1:F1}`" x2=`"{0:F1}`" y2=`"{2:F1}`" stroke=`"#a9b8c7`" stroke-width=`"1`" stroke-dasharray=`"5 6`" opacity=`"0.75`"/>" -f ($plotX+$plotW),$plotY,($plotY+$plotH)))

# 45 ms bracket.
$bracketX1 = $plotX
$bracketX2 = $plotX + $plotW / $cycleCount
$bracketY = 181
[void]$sb.AppendLine(("<line x1=`"{0:F1}`" y1=`"$bracketY`" x2=`"{1:F1}`" y2=`"$bracketY`" stroke=`"#d97706`" stroke-width=`"3`"/>" -f $bracketX1,$bracketX2))
[void]$sb.AppendLine(("<path d=`"M {0:F1} {1} l 10 -6 v 12 z`" fill=`"#d97706`"/><path d=`"M {2:F1} {1} l -10 -6 v 12 z`" fill=`"#d97706`"/>" -f $bracketX1,$bracketY,$bracketX2))
[void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"175`" class=`"callout`" text-anchor=`"middle`">約 45 ms</text>" -f (($bracketX1+$bracketX2)/2)))

for ($m = 0; $m -lt $metrics.Count; $m++) {
    $metric = $metrics[$m]
    $laneY = $plotY + $m * ($laneH + $laneGap)
    [void]$sb.AppendLine(("<rect x=`"74`" y=`"{0:F1}`" width=`"1431`" height=`"{1:F1}`" rx=`"12`" fill=`"#f4f7fa`" stroke=`"#e0e7ef`" stroke-width=`"1`"/>" -f $laneY,$laneH))
    [void]$sb.AppendLine(("<text x=`"94`" y=`"{0:F1}`" class=`"label`">{1}</text>" -f ($laneY+37),(Escape-Xml $metric.Label)))
    $raw = @($rows | ForEach-Object { [double]$_.PSObject.Properties[$metric.Name].Value })
    $values = Smooth-Values $raw 2
    $maxValue = ($values | Measure-Object -Maximum).Maximum
    if ($maxValue -le 0) { $maxValue = 1 }
    $points = [System.Text.StringBuilder]::new()
    for ($i = 0; $i -lt $values.Count; $i++) {
        $time = [double]$rows[$i].time_ms
        $x = $plotX + (($time - $startMs) / ($endMs - $startMs)) * $plotW
        $normalized = [Math]::Min(1.0, $values[$i] / $maxValue)
        $y = $laneY + $laneH - 14 - $normalized * ($laneH - 28)
        if ($i -gt 0) { [void]$points.Append(' ') }
        [void]$points.Append(("{0:F1},{1:F1}" -f $x,$y))
    }
    [void]$sb.AppendLine(("<polyline points=`"{0}`" fill=`"none`" stroke=`"{1}`" stroke-width=`"3`" stroke-linejoin=`"round`" stroke-linecap=`"round`"/>" -f $points.ToString(),$metric.Color))
    [void]$sb.AppendLine(("<text x=`"94`" y=`"{0:F1}`" class=`"small`">peak {1:F1}%</text>" -f ($laneY+64),$maxValue))
}

# Bottom axis and explanation.
[void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"{1:F1}`" class=`"axis`" text-anchor=`"start`">3.150 s</text>" -f $plotX,($plotY+$plotH+28)))
[void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"{1:F1}`" class=`"axis`" text-anchor=`"end`">3.600 s</text>" -f ($plotX+$plotW),($plotY+$plotH+28)))
[void]$sb.AppendLine('<rect x="74" y="798" width="1431" height="66" rx="14" fill="#edf3f8" stroke="#dbe4ed" stroke-width="1"/>')
[void]$sb.AppendLine('<circle cx="108" cy="831" r="7" fill="#d97706"/>')
[void]$sb.AppendLine('<text x="130" y="837" class="label">相似峰形約每 45 ms 重複，且同時出現在 compute、tensor 與 memory 指標，反映每次 training iteration 的重複工作。</text>')
[void]$sb.AppendLine('<text x="1500" y="886" class="axis" text-anchor="end">Source: analysis_cache/all_gpu_metrics_timeseries/RTX4090/C01_all_35_metrics_1ms.csv</text>')
[void]$sb.AppendLine('</svg>')

[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputSvg), $sb.ToString(), [System.Text.UTF8Encoding]::new($false))

# Render SVG to PNG through the installed PowerPoint engine.
$renderDir = Join-Path $outDir "yolo_periodicity_render"
[System.IO.Directory]::CreateDirectory($renderDir) | Out-Null
$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$presentation = $ppt.Presentations.Add()
try {
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540
    $slide = $presentation.Slides.Add(1, 12)
    $slide.Shapes.AddPicture([System.IO.Path]::GetFullPath($OutputSvg), 0, -1, 0, 0, 960, 540) | Out-Null
    $presentation.Export($renderDir, "PNG", 1600, 900)
    $rendered = Get-ChildItem -LiteralPath $renderDir -Filter '*.PNG' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $rendered) { throw "PowerPoint did not render the SVG." }
    [System.IO.File]::Copy($rendered.FullName, [System.IO.Path]::GetFullPath($OutputPng), $true)
}
finally {
    if ($presentation) { $presentation.Close() }
    if ($ppt) { $ppt.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output "SVG=$([System.IO.Path]::GetFullPath($OutputSvg))"
Write-Output "PNG=$([System.IO.Path]::GetFullPath($OutputPng))"
Write-Output "WINDOW=$startMs-$endMs ms; PERIOD=$periodMs ms; CYCLES=$cycleCount"
