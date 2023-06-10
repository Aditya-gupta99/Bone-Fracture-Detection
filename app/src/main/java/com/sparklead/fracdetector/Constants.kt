package com.sparklead.fracdetector

import android.app.Activity
import android.content.Intent
import android.provider.MediaStore

object Constants {

    const val READ_STORAGE_PERMISSION_CODE = 2
    const val PICK_IMAGE_REQUEST_CODE = 21

    fun showImageChooser(activity : Activity)
    {
        val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        activity.startActivityForResult(galleryIntent, PICK_IMAGE_REQUEST_CODE)
    }

}