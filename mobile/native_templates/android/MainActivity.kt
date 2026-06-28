package io.github.muratdelen.retinextapetum_camera

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.Executors

/** Runs RetinexTapetum locally through ONNX Runtime for Android. */
class MainActivity : FlutterActivity() {
    private val channelName = "retinex_tapetum/inference"
    private val worker = Executors.newSingleThreadExecutor()
    private val environment by lazy { OrtEnvironment.getEnvironment() }
    private var session: OrtSession? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "enhance" -> handleEnhance(call, result)
                    else -> result.notImplemented()
                }
            }
    }

    private fun handleEnhance(call: MethodCall, result: MethodChannel.Result) {
        @Suppress("UNCHECKED_CAST")
        val args = call.arguments as? Map<String, Any>
        val input = args?.get("input") as? FloatArray
        if (input == null) {
            result.error("invalid_input", "A Float32 input tensor is required.", null)
            return
        }
        if (input.size != INPUT_ELEMENTS) {
            result.error("invalid_input", "Expected $INPUT_ELEMENTS floats, received ${input.size}.", null)
            return
        }

        worker.execute {
            try {
                val output = runModel(input)
                runOnUiThread { result.success(output) }
            } catch (error: Exception) {
                runOnUiThread {
                    result.error("inference_failed", error.message ?: "ONNX inference failed.", null)
                }
            }
        }
    }

    @Synchronized
    private fun getSession(): OrtSession {
        session?.let { return it }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = environment.createSession(copyModelToFilesDir(applicationContext).absolutePath, options)
        return session!!
    }

    private fun runModel(input: FloatArray): FloatArray {
        val inputBuffer = ByteBuffer.allocateDirect(input.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        inputBuffer.put(input)
        inputBuffer.rewind()

        val tensor = OnnxTensor.createTensor(environment, inputBuffer, longArrayOf(1, 3, 512, 512))
        tensor.use {
            val activeSession = getSession()
            val inputName = activeSession.inputNames.first()
            val result = activeSession.run(mapOf(inputName to tensor))
            result.use {
                val output = result[0] as OnnxTensor
                val outputBuffer: FloatBuffer = output.floatBuffer
                    ?: throw IllegalStateException("The model did not return a float tensor.")
                val values = FloatArray(outputBuffer.remaining())
                outputBuffer.get(values)
                return values
            }
        }
    }

    private fun copyModelToFilesDir(context: Context): File {
        val target = File(context.filesDir, "retinex_tapetum_512.onnx")
        if (target.exists() && target.length() > 0) return target

        try {
            context.assets.open("flutter_assets/assets/models/retinex_tapetum_512.onnx").use { input ->
                FileOutputStream(target).use { output -> input.copyTo(output) }
            }
        } catch (error: Exception) {
            throw IllegalStateException(
                "The ONNX model asset is missing. Add mobile/assets/models/retinex_tapetum_512.onnx before running inference.",
                error,
            )
        }
        return target
    }

    override fun onDestroy() {
        session?.close()
        worker.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val INPUT_ELEMENTS = 1 * 3 * 512 * 512
    }
}
