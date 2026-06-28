from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANDROID = ROOT / "android"
APP = ANDROID / "app"
NATIVE = ROOT / "native_templates" / "android"
PACKAGE_DIR = APP / "src" / "main" / "kotlin" / "io" / "github" / "muratdelen" / "retinextapetum_camera"


def patch_gradle(path: Path) -> None:
    text = path.read_text()
    dependency = 'implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")'
    if dependency not in text:
        marker = "dependencies {"
        if marker in text:
            text = text.replace(marker, f"{marker}\n    {dependency}", 1)
        else:
            text += f"\n\ndependencies {{\n    {dependency}\n}}\n"
    text = text.replace("minSdk = flutter.minSdkVersion", "minSdk = 24")
    text = text.replace("minSdkVersion flutter.minSdkVersion", "minSdkVersion 24")
    path.write_text(text)


def main() -> None:
    if not ANDROID.exists():
        raise SystemExit("The Android host project is missing. Run flutter create first.")

    gradle_kts = APP / "build.gradle.kts"
    gradle_groovy = APP / "build.gradle"
    if gradle_kts.exists():
        patch_gradle(gradle_kts)
    elif gradle_groovy.exists():
        text = gradle_groovy.read_text()
        dependency = "implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.26.0'"
        if dependency not in text:
            text = text.replace("dependencies {", f"dependencies {{\n    {dependency}", 1)
        text = text.replace("minSdkVersion flutter.minSdkVersion", "minSdkVersion 24")
        gradle_groovy.write_text(text)
    else:
        raise SystemExit("No Android app Gradle file was found.")

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NATIVE / "MainActivity.kt", PACKAGE_DIR / "MainActivity.kt")
    shutil.copy2(NATIVE / "AndroidManifest.xml", APP / "src" / "main" / "AndroidManifest.xml")
    print("Android configured with ONNX Runtime and minSdk 24.")


if __name__ == "__main__":
    main()
