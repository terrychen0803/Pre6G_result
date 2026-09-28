param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

function Pt([double]$Inches) { return $Inches * 72.0 }
function Rgb([int]$R, [int]$G, [int]$B) { return $R + 256 * $G + 65536 * $B }

$root = Split-Path -Parent $PSScriptRoot
if (-not $OutputPath) {
    $OutputPath = Join-Path $root "analysis_outputs\presentations\ridge_profiling_overhead_correction_2slides.pptx"
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$previewDir = Join-Path ([System.IO.Path]::GetDirectoryName($OutputPath)) "ridge_overhead_preview"
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($OutputPath)) | Out-Null
[System.IO.Directory]::CreateDirectory($previewDir) | Out-Null

$navy = Rgb 9 18 31
$panel = Rgb 23 38 57
$panel2 = Rgb 31 50 72
$white = Rgb 246 249 252
$muted = Rgb 168 186 202
$teal = Rgb 40 203 185
$cyan = Rgb 77 166 255
$amber = Rgb 255 183 76
$red = Rgb 255 106 112
$green = Rgb 103 220 148
$font = "Microsoft JhengHei"

function Add-Box($slide, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [double]$radius = 0) {
    $shapeType = if ($radius -gt 0) { 5 } else { 1 }
    $shape = $slide.Shapes.AddShape($shapeType, (Pt $x), (Pt $y), (Pt $w), (Pt $h))
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Line.Visible = 0
    return $shape
}

function Add-Text($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h,
                  [double]$size, [int]$color, [bool]$bold = $false, [int]$align = 1) {
    $shape = $slide.Shapes.AddTextbox(1, (Pt $x), (Pt $y), (Pt $w), (Pt $h))
    $shape.TextFrame2.MarginLeft = Pt 0.02
    $shape.TextFrame2.MarginRight = Pt 0.02
    $shape.TextFrame2.MarginTop = Pt 0.01
    $shape.TextFrame2.MarginBottom = Pt 0.01
    $shape.TextFrame2.WordWrap = -1
    $shape.TextFrame2.TextRange.Text = $text
    $shape.TextFrame2.TextRange.Font.Name = $font
    $shape.TextFrame2.TextRange.Font.Size = $size
    $shape.TextFrame2.TextRange.Font.Bold = if ($bold) { -1 } else { 0 }
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $align
    return $shape
}

function Add-Line($slide, [double]$x1, [double]$y1, [double]$x2, [double]$y2, [int]$color, [double]$weight = 1) {
    $line = $slide.Shapes.AddLine((Pt $x1), (Pt $y1), (Pt $x2), (Pt $y2))
    $line.Line.ForeColor.RGB = $color
    $line.Line.Weight = $weight
    return $line
}

function Add-Title($slide, [string]$kicker, [string]$title, [string]$page) {
    Add-Text $slide $kicker.ToUpper() 0.60 0.27 5.5 0.25 9 $teal $true | Out-Null
    Add-Text $slide $title 0.58 0.55 11.6 0.50 24 $white $true | Out-Null
    Add-Box $slide 0.60 1.12 0.95 0.035 $teal | Out-Null
    Add-Text $slide $page 12.28 0.30 0.45 0.24 9 $muted $true 3 | Out-Null
}

function Add-Footer($slide, [string]$section) {
    Add-Text $slide "YOLO26｜Nsight GPU Metrics profiling overhead correction" 0.60 7.18 7.3 0.16 7.5 $muted | Out-Null
    Add-Text $slide $section 10.45 7.18 2.28 0.16 7.5 $muted $true 3 | Out-Null
}

function Add-FlowNode($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [int]$accent) {
    Add-Box $slide $x $y $w $h $fill 1 | Out-Null
    Add-Box $slide $x $y 0.055 $h $accent | Out-Null
    Add-Text $slide $text ($x + 0.13) ($y + 0.13) ($w - 0.23) ($h - 0.22) 12.5 $white $true 2 | Out-Null
}

function Add-MetricCard($slide, [string]$value, [string]$label, [string]$note,
                        [double]$x, [double]$y, [double]$w, [double]$h, [int]$accent) {
    Add-Box $slide $x $y $w $h $panel 1 | Out-Null
    Add-Box $slide $x $y 0.06 $h $accent | Out-Null
    Add-Text $slide $value ($x + 0.20) ($y + 0.13) ($w - 0.30) 0.42 22 $accent $true | Out-Null
    Add-Text $slide $label ($x + 0.20) ($y + 0.58) ($w - 0.30) 0.27 10.5 $white $true | Out-Null
    Add-Text $slide $note ($x + 0.20) ($y + 0.91) ($w - 0.30) ($h - 0.98) 8.5 $muted | Out-Null
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$presentation = $ppt.Presentations.Add()
$presentation.PageSetup.SlideWidth = Pt 13.333
$presentation.PageSetup.SlideHeight = Pt 7.5

try {
    # Slide 1 — method
    $slide = $presentation.Slides.Add(1, 12)
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.Solid()
    $slide.Background.Fill.ForeColor.RGB = $navy
    Add-Title $slide "METHOD" "用 Ridge Regression 修正 Profiling Overhead" "01"

    Add-Text $slide "問題" 0.72 1.42 1.0 0.25 12 $teal $true | Out-Null
    Add-Text $slide "Profiler 量到的 iteration period 含有量測擾動；直接把它當 dry runtime，會系統性高估執行時間。" 0.72 1.77 5.15 0.82 17 $white $true | Out-Null
    Add-Box $slide 0.72 2.77 5.15 1.08 $panel 1 | Out-Null
    Add-Text $slide "觀測 overhead ratio" 0.96 2.98 1.75 0.23 10 $muted $true | Out-Null
    Add-Text $slide "r = tprofile / tdry" 2.72 2.91 2.70 0.38 20 $amber $true 2 | Out-Null
    Add-Text $slide "nsys_gpu_metrics：48 組平均 ratio 1.3309（約 +33%）" 0.96 3.39 4.48 0.22 10.5 $white | Out-Null

    Add-Text $slide "Ridge 校正流程" 6.30 1.42 2.2 0.25 12 $teal $true | Out-Null
    Add-FlowNode $slide "Marker-free profiling`nperiod + GPU behavior" 6.30 1.82 2.05 1.05 $panel2 $cyan
    Add-Text $slide "→" 8.42 2.13 0.35 0.28 17 $muted $true 2 | Out-Null
    Add-FlowNode $slide "Standardize`n+ Ridge(α)" 8.82 1.82 1.65 1.05 $panel2 $teal
    Add-Text $slide "→" 10.54 2.13 0.35 0.28 17 $muted $true 2 | Out-Null
    Add-FlowNode $slide "預測 log(r)`n並限制 r̂ ≥ 1" 10.94 1.82 1.67 1.05 $panel2 $amber

    Add-Box $slide 6.30 3.13 6.31 1.12 $panel 1 | Out-Null
    Add-Text $slide "推論時的校正公式" 6.57 3.34 1.85 0.22 10 $muted $true | Out-Null
    Add-Text $slide "t̂dry = tdetected / max(1, exp(ẑ))" 7.92 3.29 4.42 0.36 16.5 $teal $true 2 | Out-Null
    Add-Text $slide "輸出同時包含 predicted overhead (%) 與 corrected dry runtime (ms)" 6.57 3.76 5.72 0.22 9.5 $white | Out-Null

    Add-Box $slide 0.72 4.36 11.89 2.25 $panel 1 | Out-Null
    Add-Text $slide "資料契約：訓練有 label，部署不偷看 baseline" 0.98 4.63 5.6 0.30 15 $white $true | Out-Null
    Add-Box $slide 0.98 5.13 3.45 1.13 $panel2 1 | Out-Null
    Add-Text $slide "DEPLOYMENT INPUT" 1.18 5.31 2.65 0.20 9 $teal $true | Out-Null
    Add-Text $slide "period、detector quality、GPU cycle behavior" 1.18 5.61 2.88 0.42 11 $white $true | Out-Null
    Add-Box $slide 4.66 5.13 3.45 1.13 $panel2 1 | Out-Null
    Add-Text $slide "LABEL ONLY" 4.86 5.31 2.65 0.20 9 $amber $true | Out-Null
    Add-Text $slide "profile steady window + 3 次 dry-run median" 4.86 5.61 2.88 0.42 11 $white $true | Out-Null
    Add-Box $slide 8.34 5.13 3.96 1.13 $panel2 1 | Out-Null
    Add-Text $slide "EVALUATION" 8.54 5.31 2.65 0.20 9 $cyan $true | Out-Null
    Add-Text $slide "每張 GPU 各自做 leave-one-workload-out；inner 5-fold 選 α / feature family" 8.54 5.56 3.38 0.52 10.5 $white $true | Out-Null
    Add-Footer $slide "RIDGE OVERHEAD｜METHOD"

    # Slide 2 — results and deployment
    $slide = $presentation.Slides.Add(2, 12)
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.Solid()
    $slide.Background.Fill.ForeColor.RGB = $navy
    Add-Title $slide "RESULTS & DEPLOYMENT" "Held-out 結果：校正後 dry runtime MAPE 降至 4.7–6.7%" "02"

    Add-Text $slide "最佳固定 Ridge family（每張 GPU 24 個 held-out workloads）" 0.72 1.38 6.3 0.26 12 $teal $true | Out-Null
    Add-MetricCard $slide "8.55 pp" "RTX4090｜overhead MAE" "period_only_ridge；dry runtime MAPE 6.72%" 0.72 1.78 2.72 1.42 $cyan
    Add-MetricCard $slide "5.40 pp" "RTX5090｜overhead MAE" "period_full_ridge；dry runtime MAPE 4.70%" 3.63 1.78 2.72 1.42 $teal

    Add-Box $slide 0.72 3.47 5.63 2.46 $panel 1 | Out-Null
    Add-Text $slide "相較 constant median ratio" 0.98 3.71 3.5 0.26 13 $white $true | Out-Null
    Add-Text $slide "RTX4090" 1.02 4.20 1.12 0.22 10 $muted $true | Out-Null
    Add-Box $slide 2.13 4.18 2.60 0.24 $panel2 | Out-Null
    Add-Box $slide 2.13 4.18 1.53 0.24 $cyan | Out-Null
    Add-Text $slide "11.45% → 6.72%  （−41%）" 4.83 4.14 1.20 0.32 10.5 $cyan $true 3 | Out-Null
    Add-Text $slide "RTX5090" 1.02 4.78 1.12 0.22 10 $muted $true | Out-Null
    Add-Box $slide 2.13 4.76 2.60 0.24 $panel2 | Out-Null
    Add-Box $slide 2.13 4.76 1.06 0.24 $teal | Out-Null
    Add-Text $slide "11.58% → 4.70%  （−59%）" 4.83 4.72 1.20 0.32 10.5 $teal $true 3 | Out-Null
    Add-Text $slide "bar 長度代表校正後 / 未校正 proxy 的相對誤差；數字為 dry runtime MAPE" 0.98 5.38 4.90 0.28 9 $muted | Out-Null

    Add-Text $slide "目前 production fit" 6.78 1.38 2.8 0.26 12 $teal $true | Out-Null
    Add-Box $slide 6.78 1.78 5.83 2.15 $panel 1 | Out-Null
    Add-Text $slide "GPU" 7.04 2.03 0.85 0.20 9 $muted $true | Out-Null
    Add-Text $slide "Feature family" 8.08 2.03 1.63 0.20 9 $muted $true | Out-Null
    Add-Text $slide "α" 9.93 2.03 0.45 0.20 9 $muted $true 2 | Out-Null
    Add-Text $slide "CV MAE" 10.55 2.03 0.85 0.20 9 $muted $true 2 | Out-Null
    Add-Text $slide "Features" 11.55 2.03 0.77 0.20 9 $muted $true 2 | Out-Null
    Add-Line $slide 7.02 2.35 12.33 2.35 $panel2 1 | Out-Null
    Add-Text $slide "RTX4090" 7.04 2.55 0.88 0.25 11 $cyan $true | Out-Null
    Add-Text $slide "period_only" 8.08 2.55 1.63 0.25 11 $white $true | Out-Null
    Add-Text $slide "0.1" 9.93 2.55 0.45 0.25 11 $white $true 2 | Out-Null
    Add-Text $slide "8.95 pp" 10.55 2.55 0.85 0.25 11 $white $true 2 | Out-Null
    Add-Text $slide "1" 11.55 2.55 0.77 0.25 11 $white $true 2 | Out-Null
    Add-Line $slide 7.02 2.93 12.33 2.93 $panel2 1 | Out-Null
    Add-Text $slide "RTX5090" 7.04 3.13 0.88 0.25 11 $teal $true | Out-Null
    Add-Text $slide "behavior_only" 8.08 3.13 1.63 0.25 11 $white $true | Out-Null
    Add-Text $slide "1.0" 9.93 3.13 0.45 0.25 11 $white $true 2 | Out-Null
    Add-Text $slide "4.77 pp" 10.55 3.13 0.85 0.25 11 $white $true 2 | Out-Null
    Add-Text $slide "83" 11.55 3.13 0.77 0.25 11 $white $true 2 | Out-Null

    Add-Box $slide 6.78 4.18 5.83 1.75 $panel 1 | Out-Null
    Add-Text $slide "如何使用" 7.04 4.42 1.35 0.24 13 $white $true | Out-Null
    Add-Text $slide "1  輸入同裝置的 marker-free feature CSV" 7.04 4.86 4.90 0.24 10.5 $white | Out-Null
    Add-Text $slide "2  載入對應 RTX4090 / RTX5090 joblib" 7.04 5.19 4.90 0.24 10.5 $white | Out-Null
    Add-Text $slide "3  產生 predicted overhead 與 corrected dry runtime" 7.04 5.52 4.90 0.24 10.5 $white | Out-Null

    Add-Box $slide 0.72 6.18 11.89 0.73 (Rgb 62 40 44) 1 | Out-Null
    Add-Text $slide "適用邊界" 0.96 6.40 1.00 0.22 10 $red $true | Out-Null
    Add-Text $slide "僅驗證 YOLO26、已知 RTX4090/5090、nsys_gpu_metrics；每個 profiler/workload 目前只有 1 次 profile，不能外推到新 GPU 或其他 profiler mode。" 2.02 6.34 10.15 0.34 10.5 $white $true | Out-Null
    Add-Footer $slide "RIDGE OVERHEAD｜RESULTS"

    $presentation.SaveAs($OutputPath, 24)
    $presentation.Export($previewDir, "PNG", 1600, 900)
    Write-Output "PPTX=$OutputPath"
    Write-Output "PREVIEW=$previewDir"
    Write-Output "SLIDES=$($presentation.Slides.Count)"
}
finally {
    if ($presentation) { $presentation.Close() }
    if ($ppt) { $ppt.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
