param(
    [string]$OutputSvg = "",
    [string]$OutputPng = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$experimentDir = Join-Path $root "analysis_outputs\model_experiments\yolo_single_device_overhead"
$samplesPath = Join-Path $experimentDir "samples.csv"
$metricsPath = Join-Path $experimentDir "metrics.csv"
$predictionsPath = Join-Path $experimentDir "predictions.csv"
$outDir = Join-Path $root "analysis_outputs\figures"
[System.IO.Directory]::CreateDirectory($outDir) | Out-Null
if (-not $OutputSvg) { $OutputSvg = Join-Path $outDir "ridge_overhead_correction_actual_results.svg" }
if (-not $OutputPng) { $OutputPng = Join-Path $outDir "ridge_overhead_correction_actual_results.png" }

$samples = Import-Csv -LiteralPath $samplesPath
$metrics = Import-Csv -LiteralPath $metricsPath
$nestedPredictions = Import-Csv -LiteralPath $predictionsPath | Where-Object model -eq "nested_selected_ridge"
$deviceColors = @{ RTX4090 = "#348fe2"; RTX5090 = "#16a897" }

$summary = foreach ($device in "RTX4090", "RTX5090") {
    $group = @($samples | Where-Object device_id -eq $device)
    $rawErrors = @($group | ForEach-Object {
        [Math]::Abs(([double]$_.detected_period_ms - [double]$_.actual_dry_runtime_ms) / [double]$_.actual_dry_runtime_ms) * 100
    })
    $overheads = @($group | ForEach-Object { [double]$_.actual_overhead_percent } | Sort-Object)
    $metric = $metrics | Where-Object { $_.device_id -eq $device -and $_.model -eq "nested_selected_ridge" }
    [pscustomobject]@{
        Device = $device
        N = $group.Count
        RawMape = ($rawErrors | Measure-Object -Average).Average
        CorrectedMape = [double]$metric.estimated_dry_runtime_mape_percent
        OverheadMae = [double]$metric.overhead_mae_pp
        OverheadMean = ($overheads | Measure-Object -Average).Average
        OverheadMedian = ($overheads[11] + $overheads[12]) / 2
    }
}

$pairs = foreach ($sample in $samples) {
    $prediction = $nestedPredictions | Where-Object {
        $_.device_id -eq $sample.device_id -and $_.workload_id -eq $sample.workload_id
    }
    [pscustomobject]@{
        Device = $sample.device_id
        Workload = $sample.workload_id
        RawApe = [Math]::Abs(([double]$sample.detected_period_ms - [double]$sample.actual_dry_runtime_ms) / [double]$sample.actual_dry_runtime_ms) * 100
        CorrectedApe = [double]$prediction.dry_runtime_ape_percent
    }
}
$improvedCount = @($pairs | Where-Object { $_.CorrectedApe -lt $_.RawApe }).Count

function Esc([string]$text) { return [System.Security.SecurityElement]::Escape($text) }
function XMap([double]$value, [double]$x, [double]$width, [double]$max) { return $x + ($value / $max) * $width }
function YMap([double]$value, [double]$y, [double]$height, [double]$max) { return $y + $height - ($value / $max) * $height }

$W = 1600
$H = 900
$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine('<?xml version="1.0" encoding="UTF-8"?>')
[void]$sb.AppendLine("<svg xmlns=`"http://www.w3.org/2000/svg`" width=`"$W`" height=`"$H`" viewBox=`"0 0 $W $H`">")
[void]$sb.AppendLine('<rect width="1600" height="900" fill="#ffffff"/>')
[void]$sb.AppendLine('<style>text{font-family:"Microsoft JhengHei","Noto Sans TC",Arial,sans-serif}.title{font-size:38px;font-weight:700;fill:#172033}.subtitle{font-size:17px;fill:#607184}.kicker{font-size:14px;font-weight:700;fill:#087f73;letter-spacing:1.5px}.panelTitle{font-size:20px;font-weight:700;fill:#172033}.body{font-size:16px;fill:#27364a}.small{font-size:13px;fill:#667789}.axis{font-size:12px;fill:#718296}.value{font-size:22px;font-weight:700}.cardValue{font-size:24px;font-weight:700;fill:#172033}.cardLabel{font-size:13px;fill:#667789}</style>')

[void]$sb.AppendLine('<text x="74" y="55" class="kicker">PROFILING OVERHEAD CORRECTION · ACTUAL EXPERIMENT DATA</text>')
[void]$sb.AppendLine('<text x="74" y="108" class="title">Ridge 修正後，dry-runtime 預測誤差下降 73–83%</text>')
[void]$sb.AppendLine('<text x="74" y="142" class="subtitle">YOLO26 · Nsight GPU Metrics · 每張 GPU 24 個 held-out workloads · nested leave-one-workload-out</text>')

# Actual overhead summary cards.
$cardX = @(74, 790)
for ($i = 0; $i -lt $summary.Count; $i++) {
    $s = $summary[$i]
    $x = $cardX[$i]
    $color = $deviceColors[$s.Device]
    [void]$sb.AppendLine(("<rect x=`"{0}`" y=`"165`" width=`"650`" height=`"88`" rx=`"14`" fill=`"#f4f7fa`" stroke=`"#dce5ed`"/>" -f $x))
    [void]$sb.AppendLine(("<rect x=`"{0}`" y=`"165`" width=`"7`" height=`"88`" rx=`"3`" fill=`"{1}`"/>" -f $x,$color))
    [void]$sb.AppendLine(("<text x=`"{0}`" y=`"196`" class=`"panelTitle`" fill=`"{1}`">{2}</text>" -f ($x+24),$color,$s.Device))
    [void]$sb.AppendLine(("<text x=`"{0}`" y=`"231`" class=`"cardLabel`">實際 profiling overhead</text>" -f ($x+24)))
    [void]$sb.AppendLine(("<text x=`"{0}`" y=`"231`" class=`"body`">平均 <tspan font-weight=`"700`">{1:F2}%</tspan>　中位數 <tspan font-weight=`"700`">{2:F2}%</tspan>　修正模型 MAE <tspan font-weight=`"700`" fill=`"{3}`">{4:F2} pp</tspan></text>" -f ($x+187),$s.OverheadMean,$s.OverheadMedian,$color,$s.OverheadMae))
}

# Left panel: before/after dumbbell.
[void]$sb.AppendLine('<rect x="74" y="278" width="650" height="470" rx="18" fill="#fbfcfe" stroke="#dce5ed"/>')
[void]$sb.AppendLine('<text x="102" y="318" class="panelTitle">平均 dry-runtime 誤差：修正前 vs 修正後</text>')
[void]$sb.AppendLine('<text x="102" y="345" class="small">修正前：detected period 直接對 dry-run GT；修正後：nested-selected Ridge</text>')
$axisX = 210.0
$axisW = 450.0
$axisMax = 35.0
$axisY = 682.0
foreach ($tick in 0, 10, 20, 30) {
    $x = XMap $tick $axisX $axisW $axisMax
    [void]$sb.AppendLine(("<line x1=`"{0:F1}`" y1=`"382`" x2=`"{0:F1}`" y2=`"{1}`" stroke=`"#e2e8ef`" stroke-width=`"1`"/>" -f $x,$axisY))
    [void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"706`" class=`"axis`" text-anchor=`"middle`">{1}%</text>" -f $x,$tick))
}
$rowY = @(455.0, 585.0)
for ($i = 0; $i -lt $summary.Count; $i++) {
    $s = $summary[$i]
    $y = $rowY[$i]
    $color = $deviceColors[$s.Device]
    $rawX = XMap $s.RawMape $axisX $axisW $axisMax
    $correctedX = XMap $s.CorrectedMape $axisX $axisW $axisMax
    $reduction = (1 - $s.CorrectedMape / $s.RawMape) * 100
    [void]$sb.AppendLine(("<text x=`"102`" y=`"{0:F1}`" class=`"panelTitle`" fill=`"{1}`">{2}</text>" -f ($y+6),$color,$s.Device))
    [void]$sb.AppendLine(("<line x1=`"{0:F1}`" y1=`"{1:F1}`" x2=`"{2:F1}`" y2=`"{1:F1}`" stroke=`"#aebbc8`" stroke-width=`"7`" stroke-linecap=`"round`"/>" -f $correctedX,$y,$rawX))
    [void]$sb.AppendLine(("<circle cx=`"{0:F1}`" cy=`"{1:F1}`" r=`"11`" fill=`"#f59e0b`"/><circle cx=`"{2:F1}`" cy=`"{1:F1}`" r=`"12`" fill=`"{3}`"/>" -f $rawX,$y,$correctedX,$color))
    [void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"{1:F1}`" class=`"small`" text-anchor=`"middle`">修正前 {2:F2}%</text>" -f $rawX,($y-24),$s.RawMape))
    [void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"{1:F1}`" class=`"small`" text-anchor=`"middle`">修正後 {2:F2}%</text>" -f $correctedX,($y+37),$s.CorrectedMape))
    [void]$sb.AppendLine(("<rect x=`"{0:F1}`" y=`"{1:F1}`" width=`"96`" height=`"28`" rx=`"14`" fill=`"{2}`" opacity=`"0.12`"/>" -f (($correctedX+$rawX)/2-48),($y-14),$color))
    [void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"{1:F1}`" class=`"small`" text-anchor=`"middle`" font-weight=`"700`" fill=`"{2}`">−{3:F0}%</text>" -f (($correctedX+$rawX)/2),($y+5),$color,$reduction))
}
[void]$sb.AppendLine('<circle cx="114" cy="728" r="6" fill="#f59e0b"/><text x="128" y="733" class="small">修正前</text><circle cx="210" cy="728" r="6" fill="#348fe2"/><text x="224" y="733" class="small">Ridge 修正後</text>')

# Right panel: per-workload paired scatter.
[void]$sb.AppendLine('<rect x="790" y="278" width="735" height="470" rx="18" fill="#fbfcfe" stroke="#dce5ed"/>')
[void]$sb.AppendLine('<text x="818" y="318" class="panelTitle">逐 workload 誤差（48 組 OOF 預測）</text>')
[void]$sb.AppendLine(("<text x=`"818`" y=`"345`" class=`"small`">{0}/48 個點位於改善區（對角線下方）</text>" -f $improvedCount))
$plotX = 870.0
$plotY = 370.0
$plotW = 590.0
$plotH = 315.0
$scatterMax = 60.0
foreach ($tick in 0, 15, 30, 45, 60) {
    $x = XMap $tick $plotX $plotW $scatterMax
    $y = YMap $tick $plotY $plotH $scatterMax
    [void]$sb.AppendLine(("<line x1=`"{0:F1}`" y1=`"{1}`" x2=`"{0:F1}`" y2=`"{2}`" stroke=`"#e3e9ef`"/>" -f $x,$plotY,($plotY+$plotH)))
    [void]$sb.AppendLine(("<line x1=`"{0}`" y1=`"{1:F1}`" x2=`"{2}`" y2=`"{1:F1}`" stroke=`"#e3e9ef`"/>" -f $plotX,$y,($plotX+$plotW)))
    [void]$sb.AppendLine(("<text x=`"{0:F1}`" y=`"706`" class=`"axis`" text-anchor=`"middle`">{1}</text>" -f $x,$tick))
    [void]$sb.AppendLine(("<text x=`"850`" y=`"{0:F1}`" class=`"axis`" text-anchor=`"end`">{1}</text>" -f ($y+4),$tick))
}
[void]$sb.AppendLine(("<path d=`"M {0} {1} L {2} {3}`" stroke=`"#9caaba`" stroke-width=`"2`" stroke-dasharray=`"7 6`"/>" -f $plotX,($plotY+$plotH),($plotX+$plotW),$plotY))
[void]$sb.AppendLine('<text x="1438" y="388" class="axis" text-anchor="end">無改善線 y = x</text>')
foreach ($pair in $pairs) {
    $x = XMap $pair.RawApe $plotX $plotW $scatterMax
    $y = YMap $pair.CorrectedApe $plotY $plotH $scatterMax
    $color = $deviceColors[$pair.Device]
    [void]$sb.AppendLine(("<circle cx=`"{0:F1}`" cy=`"{1:F1}`" r=`"6.2`" fill=`"{2}`" fill-opacity=`"0.82`" stroke=`"#ffffff`" stroke-width=`"1.5`"><title>{3} {4}: raw {5:F2}%, corrected {6:F2}%</title></circle>" -f $x,$y,$color,$pair.Device,$pair.Workload,$pair.RawApe,$pair.CorrectedApe))
}
[void]$sb.AppendLine('<text x="1165" y="731" class="small" text-anchor="middle">修正前 APE (%)</text>')
[void]$sb.AppendLine('<text x="808" y="530" class="small" text-anchor="middle" transform="rotate(-90 808 530)">修正後 APE (%)</text>')
[void]$sb.AppendLine('<circle cx="1260" cy="715" r="6" fill="#348fe2"/><text x="1273" y="720" class="small">RTX4090</text><circle cx="1360" cy="715" r="6" fill="#16a897"/><text x="1373" y="720" class="small">RTX5090</text>')

# Interpretation and sources.
[void]$sb.AppendLine('<rect x="74" y="776" width="1451" height="70" rx="14" fill="#edf5f4" stroke="#d4e8e5"/>')
[void]$sb.AppendLine('<circle cx="106" cy="811" r="7" fill="#087f73"/>')
[void]$sb.AppendLine('<text x="128" y="817" class="body" font-weight="700">結論：直接使用 profiled period 會把約 33% 的量測 overhead 當成真實 runtime；Ridge 校正後，兩張 GPU 的平均誤差都降至 8% 以下。</text>')
[void]$sb.AppendLine('<text x="1525" y="881" class="axis" text-anchor="end">Sources: samples.csv · metrics.csv · predictions.csv｜model: nested_selected_ridge</text>')
[void]$sb.AppendLine('</svg>')

[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputSvg), $sb.ToString(), [System.Text.UTF8Encoding]::new($false))

# Render the vector figure to PNG with the installed PowerPoint engine.
$renderDir = Join-Path $outDir "overhead_results_render"
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
foreach ($s in $summary) {
    Write-Output ("{0}: raw={1:F2}%, corrected={2:F2}%, overhead_mean={3:F2}%, overhead_median={4:F2}%, overhead_MAE={5:F2}pp" -f $s.Device,$s.RawMape,$s.CorrectedMape,$s.OverheadMean,$s.OverheadMedian,$s.OverheadMae)
}
