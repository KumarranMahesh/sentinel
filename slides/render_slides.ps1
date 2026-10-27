# Render each slide of the deck to PNG with PowerPoint (for visual QA).
#   powershell -ExecutionPolicy Bypass -File slides\render_slides.ps1 [-Deck path] [-Out dir]
param(
  [string]$Deck = "$PSScriptRoot\Sentinel_Challenge2.pptx",
  [string]$Out = "$PSScriptRoot\render"
)
New-Item -ItemType Directory -Force $Out | Out-Null
Get-ChildItem $Out -Filter *.png | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
  $pres = $app.Presentations.Open((Resolve-Path $Deck).Path, $true, $false, $false)
  $i = 1
  foreach ($s in $pres.Slides) {
    $s.Export((Join-Path $Out ("slide-{0:D2}.png" -f $i)), "PNG", 1600, 900)
    $i++
  }
  $pres.Close()
} finally {
  $app.Quit()
}
Get-ChildItem $Out -Filter *.png | ForEach-Object { $_.FullName }
