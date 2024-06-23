package com.example.imageclassification;

import org.tensorflow.lite.support.image.TensorImage;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.SystemClock;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.example.imageclassification.ml.Mobilenetv2;
import com.example.imageclassification.ml.MobilevitS;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;

public class MainActivity extends AppCompatActivity {

    Button selectBtn, predictBtn, captureBtn;
    TextView result, inftime;
    ImageView imageView;
    Bitmap bitmap;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // permission
        getPermission();

        String[] labels = new String[1001];
        int cnt = 0;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line = bufferedReader.readLine();
            while (line != null) {
                labels[cnt] = line;
                cnt++;
                line = bufferedReader.readLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        selectBtn = findViewById(R.id.selectBtn);
        predictBtn = findViewById(R.id.predictBtn);
        captureBtn = findViewById(R.id.captureBtn);
        result = findViewById(R.id.result);
        inftime = findViewById(R.id.inftime);
        imageView = findViewById(R.id.imageView);

        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });

        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);
            }
        });

//      Mobilenetv2
//        predictBtn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                try {
//                    if (bitmap == null) {
//                        // Bitmap이 유효하지 않은 경우
//                        Toast.makeText(MainActivity.this, "이미지를 먼저 선택해주세요.", Toast.LENGTH_SHORT).show();
//                        return;
//                    }
//
//                    Mobilenetv2 model = Mobilenetv2.newInstance(MainActivity.this);
//
//                    // Resize bitmap to 256x256
//                    Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
//
//                    // Create TensorBuffer for input data
//                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
//
//                    // Set input data directly into TensorBuffer using ByteBuffer
//                    int[] intValues = new int[224 * 224];
//                    resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());
//
//                    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4); // 1 image, 256x256 resolution, 3 channels (RGB), 4 bytes per float
//                    byteBuffer.order(ByteOrder.nativeOrder());
//                    byteBuffer.rewind();
//
//                    // Convert bitmap data to float values and write to ByteBuffer
//                    for (int pixelValue : intValues) {
//                        byteBuffer.putFloat(Color.red(pixelValue) / 255.0f);
//                        byteBuffer.putFloat(Color.green(pixelValue) / 255.0f);
//                        byteBuffer.putFloat(Color.blue(pixelValue) / 255.0f);
//                    }
//
//                    // Load ByteBuffer into TensorBuffer
//                    inputFeature0.loadBuffer(byteBuffer);
//
//                    // Measure inference time
//                    long startTime = SystemClock.elapsedRealtime();
//
//                    // Run model inference and get outputs
//                    Mobilenetv2.Outputs outputs = model.process(inputFeature0);
//                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//                    long endTime = SystemClock.elapsedRealtime();
//                    long inferenceTime = endTime - startTime;
//                    double Time = inferenceTime * 100 / 100.0;
//                    double inferenceTimeSeconds = inferenceTime / 1000.0; // Convert to seconds
//                    DecimalFormat decimalFormat = new DecimalFormat("#.##"); // Format to 2 decimal places
//                    String formattedInferenceTime = decimalFormat.format(inferenceTimeSeconds);
//                    inftime.setText("Inf. Time: " + formattedInferenceTime + " seconds");
//
//                    // Get predicted label
//                    int predictedIndex = getMax(outputFeature0.getFloatArray());
//                    String predictedLabel = labels[predictedIndex - 1];
//
//                    // Display result
//                    result.setText(predictedLabel);
//
//                    // Release model resources
//                    model.close();
//                } catch (IOException e) {
//                    e.printStackTrace();
//                    // Handle the exception
//                }
//            }
//        });

//         MobileViT
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    if (bitmap == null) {
                        // Bitmap이 유효하지 않은 경우
                        Toast.makeText(MainActivity.this, "이미지를 먼저 선택해주세요.", Toast.LENGTH_SHORT).show();
                        return;
                    }

                    MobilevitS model = MobilevitS.newInstance(MainActivity.this);

                    // Resize bitmap to 256x256
                    Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true);

                    // Create TensorBuffer for input data
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 256, 256, 3}, DataType.FLOAT32);

                    // Set input data directly into TensorBuffer using ByteBuffer
                    int[] intValues = new int[256 * 256];
                    resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

                    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4); // 1 image, 256x256 resolution, 3 channels (RGB), 4 bytes per float
                    byteBuffer.order(ByteOrder.nativeOrder());
                    byteBuffer.rewind();

                    // Convert bitmap data to float values and write to ByteBuffer
                    for (int pixelValue : intValues) {
                        byteBuffer.putFloat(Color.red(pixelValue) / 255.0f);
                        byteBuffer.putFloat(Color.green(pixelValue) / 255.0f);
                        byteBuffer.putFloat(Color.blue(pixelValue) / 255.0f);
                    }

                    // Load ByteBuffer into TensorBuffer
                    inputFeature0.loadBuffer(byteBuffer);

                    // Measure inference time
                    long startTime = SystemClock.elapsedRealtime();

                    // Run model inference and get outputs
                    MobilevitS.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    long endTime = SystemClock.elapsedRealtime();
                    long inferenceTime = endTime - startTime;
                    double Time = inferenceTime * 100 / 100.0;
                    double inferenceTimeSeconds = inferenceTime / 1000.0; // Convert to seconds
                    DecimalFormat decimalFormat = new DecimalFormat("#.##"); // Format to 2 decimal places
                    String formattedInferenceTime = decimalFormat.format(inferenceTimeSeconds);
                    inftime.setText("Inf. Time: " + formattedInferenceTime + " seconds");

                    // Get predicted label
                    int predictedIndex = getMax(outputFeature0.getFloatArray());
                    String predictedLabel = labels[predictedIndex];

                    // Display result
                    result.setText(predictedLabel);

                    // Release model resources
                    model.close();
                } catch (IOException e) {
                    e.printStackTrace();
                    // Handle the exception
                }
            }
        });
    }

    int getMax(float[] arr){
        int max = 0;
        for(int i = 0; i < arr.length; i++){
            if(arr[i] > arr[max]) max = i;
        }
        return max;
    }

    void getPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            if(checkSelfPermission(android.Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{android.Manifest.permission.CAMERA}, 11);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode==11){
            if(grantResults.length>0){
                if(grantResults[0]!=PackageManager.PERMISSION_GRANTED){
                    this.getPermission();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode==10){
            if(data!=null){
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
        else if(requestCode==12) {
            bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}