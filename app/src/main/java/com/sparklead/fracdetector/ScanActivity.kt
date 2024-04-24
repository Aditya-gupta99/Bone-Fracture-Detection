package com.sparklead.fracdetector

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.sparklead.fracdetector.databinding.ActivityScanBinding
import com.sparklead.fracdetector.ml.Resnet50Bodyparts
import com.sparklead.fracdetector.ml.Resnet50ElbowFrac
import com.sparklead.fracdetector.ml.Resnet50HandFrac
import com.sparklead.fracdetector.ml.Resnet50ShoulderFrac
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ScanActivity : AppCompatActivity(), View.OnClickListener {

    private lateinit var binding: ActivityScanBinding
    private var mBoneImageUri: String = ""
    private lateinit var mBoneBitmap: Bitmap

    private val selectImageIntent =
        registerForActivityResult(ActivityResultContracts.GetContent()) {
            binding.ivBone.setImageURI(it)
            mBoneImageUri = it.toString()
            val image = MediaStore.Images.Media.getBitmap(
                applicationContext.contentResolver,
                Uri.parse(it.toString())
            )
            val imageSize = 224
//            mBoneBitmap=  Bitmap.createScaledBitmap(image , imageSize , imageSize , false)
            mBoneBitmap = MediaStore.Images.Media.getBitmap(
                applicationContext.contentResolver,
                Uri.parse(it.toString())
            )

        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityScanBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.ivBone.setOnClickListener(this)
        binding.btnScan.setOnClickListener(this)
        binding.fabInfo.setOnClickListener(this)
    }

    override fun onClick(v: View?) {
        if (v != null) {
            when (v.id) {
                R.id.iv_bone -> {

                    val intent =
                        Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                    if (intent.resolveActivity(packageManager) != null) {
                        selectImageIntent.launch("image/*")
                    }

                }

                R.id.btn_scan -> {
                    if (mBoneImageUri == "") {
                        Toast.makeText(this, "Please upload x-ray image", Toast.LENGTH_LONG).show()
                    } else {
                        bodyPartModel()
                    }
                }

                R.id.fab_info -> {
                    showInfo()
                }
            }
        }
    }

    private fun bodyPartModel() {
        val model = Resnet50Bodyparts.newInstance(this)


//        Log.e("Before",byteBuffer.toString())
        // Creates inputs for reference.

        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)

        val input = Bitmap.createScaledBitmap(mBoneBitmap, 224, 224, true)
        val image = TensorImage(DataType.FLOAT32)
        image.load(input)
        val byteBuffer = image.buffer
        inputFeature0.loadBuffer(byteBuffer)


        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidence = outputFeature0.floatArray
        var maxi = 0
        var maxP = 0.0f
        for (i in confidence.indices) {
            if (maxP < confidence[i]) {
                maxP = confidence[i]
                maxi = i
            }
        }
        val resArray = arrayOf("Elbow", "Hand", "Shoulder")
        val ans = resArray[maxi]
//        Toast.makeText(this,confidence[0].toString()+ " "+confidence[1].toString()+" "+confidence[2].toString(),Toast.LENGTH_LONG).show()
//        Toast.makeText(this,ans,Toast.LENGTH_SHORT).show()
//        binding.tvOutput.text = "Type : $ans"
        if (maxi == 0) {
            elbowModel()
        } else if (maxi == 1) {
            handModel()
        } else {
            shoulderModel()
        }

        // Releases model resources if no longer used.
        model.close()
    }


    private fun elbowModel() {
        val model = Resnet50ElbowFrac.newInstance(this)

        // Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        val image = TensorImage(DataType.FLOAT32)
        val input = Bitmap.createScaledBitmap(mBoneBitmap, 224, 224, true)
        image.load(input)
        val byteBuffer = image.buffer
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidence = outputFeature0.floatArray
        var maxi = 0
        var maxP = 0.0f
        for (i in confidence.indices) {
            if (maxP < confidence[i]) {
                maxP = confidence[i]
                maxi = i
            }
        }
        val resArray = arrayOf("fractured", "normal")
        val ans = resArray[maxi]
        showResultDialog("Elbow", ans)
//        binding.tvOutputResult.text = "Result : $ans"
//        Toast.makeText(this,ans,Toast.LENGTH_LONG).show()

        // Releases model resources if no longer used.
        model.close()
    }

    private fun handModel() {
        val model = Resnet50HandFrac.newInstance(this)

// Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        val image = TensorImage(DataType.FLOAT32)
        val input = Bitmap.createScaledBitmap(mBoneBitmap, 224, 224, true)
        image.load(input)
        val byteBuffer = image.buffer
        inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidence = outputFeature0.floatArray
        var maxi = 0
        var maxP = 0.0f
        for (i in confidence.indices) {
            if (maxP < confidence[i]) {
                maxP = confidence[i]
                maxi = i
            }
        }
        val resArray = arrayOf("fractured", "normal")
        val ans = resArray[maxi]
        showResultDialog("Hand", ans)
//        binding.tvOutputResult.text = "Result : $ans"

//        Toast.makeText(this,ans,Toast.LENGTH_LONG).show()

// Releases model resources if no longer used.
        model.close()
    }

    private fun shoulderModel() {
        val model = Resnet50ShoulderFrac.newInstance(this)

// Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        val image = TensorImage(DataType.FLOAT32)
        val input = Bitmap.createScaledBitmap(mBoneBitmap, 224, 224, true)
        image.load(input)
        val byteBuffer = image.buffer
        inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidence = outputFeature0.floatArray
        var maxi = 0
        var maxP = 0.0f
        for (i in confidence.indices) {
            if (maxP < confidence[i]) {
                maxP = confidence[i]
                maxi = i
            }
        }
        val resArray = arrayOf("fractured", "normal")
        val ans = resArray[maxi]
        showResultDialog("Shoulder", ans)
//        binding.tvOutputResult.text = "Result : $ans"
//        Toast.makeText(this,ans,Toast.LENGTH_LONG).show()


// Releases model resources if no longer used.
        model.close()
    }

    private fun showResultDialog(type: String, fracture: String) {
        val dialog = BottomSheetDialog(this, R.style.BottomSheetStyle)
        val view = layoutInflater.inflate(R.layout.bottom_dialog_sheet, null)

        val tv = view.findViewById<TextView>(R.id.tv_output)
        tv.text = type

        val result = view.findViewById<TextView>(R.id.tv_output_result)
        result.text = fracture

        val btnReset = view.findViewById<ImageView>(R.id.reloadBtnDexa1)
        btnReset.setOnClickListener {
            startActivity(Intent(this, ScanActivity::class.java))
            finish()
        }
//        val btnHome = view.findViewById<ImageView>(R.id.deleteBtnDexa1)
//        btnHome.setOnClickListener {
//            startActivity(Intent(this,DashboardActivity::class.java))
//            finish()
//        }

        dialog.setCancelable(true)
        dialog.setContentView(view)
        dialog.show()
    }

    private fun showInfo() {
        val dialog = BottomSheetDialog(this, R.style.BottomSheetStyle)
        val view = layoutInflater.inflate(R.layout.bottom_info_sheet, null)

//        val btnHome = view.findViewById<ImageView>(R.id.deleteBtnDexa1)
//        btnHome.setOnClickListener {
//            startActivity(Intent(this,DashboardActivity::class.java))
//            finish()
//        }

        dialog.setCancelable(true)
        dialog.setContentView(view)
        dialog.show()
    }
}