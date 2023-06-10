package com.sparklead.fracdetector

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.sparklead.fracdetector.databinding.ActivityScanBinding
import com.sparklead.fracdetector.ml.Resnet50Bodyparts
import com.sparklead.fracdetector.ml.Resnet50ElbowFrac
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ScanActivity : AppCompatActivity(), View.OnClickListener {

    private lateinit var binding: ActivityScanBinding
    private lateinit var mBoneImageUri: String
    private lateinit var mBoneBitmap : Bitmap

    private val selectImageIntent =
        registerForActivityResult(ActivityResultContracts.GetContent()) {
            binding.ivBone.setImageURI(it)
            mBoneImageUri = it.toString()
            val image = MediaStore.Images.Media.getBitmap(applicationContext.contentResolver, Uri.parse(it.toString()))
            val imageSize = 224
//            mBoneBitmap=  Bitmap.createScaledBitmap(image , imageSize , imageSize , false)
            mBoneBitmap = MediaStore.Images.Media.getBitmap(applicationContext.contentResolver, Uri.parse(it.toString()))

        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityScanBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.ivBone.setOnClickListener(this)
        binding.btnScan.setOnClickListener(this)
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
                    bodyPartModel()
                }
            }
        }
    }

    private fun bodyPartModel() {
        val model = Resnet50Bodyparts.newInstance(this)


//        Log.e("Before",byteBuffer.toString())
        // Creates inputs for reference.

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)

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
        for (i in confidence.indices)
        {
            if(maxP<confidence[i])
            {
                maxP = confidence[i]
                maxi = i
            }
        }
        val resArray = arrayOf("Elbow", "Hand", "Shoulder")
        val ans = resArray[maxi]
//        Toast.makeText(this,confidence[0].toString()+ " "+confidence[1].toString()+" "+confidence[2].toString(),Toast.LENGTH_LONG).show()
//        Toast.makeText(this,ans,Toast.LENGTH_SHORT).show()
        binding.tvOutput.text = ans
        if(maxi == 0)
        {
            elbowModel()
        }

        // Releases model resources if no longer used.
        model.close()
    }


    private fun elbowModel(){
        val model = Resnet50ElbowFrac.newInstance(this)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
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
        for (i in confidence.indices)
        {
            if(maxP<confidence[i])
            {
                maxP = confidence[i]
                maxi = i
            }
        }
        val resArray = arrayOf("fractured", "normal")
        val ans = resArray[maxi]

//        Toast.makeText(this,ans,Toast.LENGTH_LONG).show()

        // Releases model resources if no longer used.
        model.close()
    }
}