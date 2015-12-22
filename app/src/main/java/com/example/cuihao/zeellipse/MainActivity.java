package com.example.cuihao.zeellipse;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Point;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;

import uk.co.senab.photoview.PhotoViewAttacher;

import static com.example.cuihao.zeellipse.JniWrapper.CppGetSobel;
import static com.example.cuihao.zeellipse.JniWrapper.CppPreprocess;
import static com.example.cuihao.zeellipse.JniWrapper.DynamicDilate;
import static com.example.cuihao.zeellipse.JniWrapper.DynamicErode;
import static com.example.cuihao.zeellipse.JniWrapper.EllipticalIntegrate;
import static com.example.cuihao.zeellipse.JniWrapper.EllipticalR;
import static com.example.cuihao.zeellipse.JniWrapper.GetRBox;
import static com.example.cuihao.zeellipse.JniWrapper.QuickFindCenter;

public class MainActivity extends AppCompatActivity {
    Button buttonLoad;
    TextView textViewMessage;
    TextView textViewInfo;
    ImageView imageViewDisplay;
    PhotoViewAttacher mAttacher;
    ToggleButton toggleButtonMode;
    Button buttonReset;

    static final double touchReactDiffPixel = 50.0;
    static final int gapWidthThreshold = 10;
    static final double derivative2Threshold = 1;

    Mat imColor, imGray;
    RotatedRect avgEllipse;
    ArrayList<Integer> dResults = new ArrayList<>();
    ArrayList<Point> selectPoints = new ArrayList<>();

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("Hello", "OpenCV loaded successfully");
                    //System.loadLibrary("libopencv_java3");
                    System.loadLibrary("jni_wrapper");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private void resetImage() {
        textViewInfo.setText("");
        selectPoints.clear();
        Bitmap bitmap = Bitmap.createBitmap(imColor.cols(), imColor.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imColor, bitmap);
        imageViewDisplay.setImageBitmap(bitmap);

        float scale1 = 1.0F * imageViewDisplay.getWidth() / imColor.cols(),
              scale2 = 1.0F * imageViewDisplay.getHeight() / imColor.rows();
        float scale = scale1 > scale2 ? scale2 : scale1;
        mAttacher.setScale(scale > mAttacher.getMinimumScale() ? scale : mAttacher.getMinimumScale());
    }

    private double[] secondDerivative(double[] seq) {
        double[] r = new double[seq.length];
        for (int i=3; i<r.length-3; i++)
            r[i] = (seq[i+3] - 2*seq[i] + seq[i-3])/4.0;
        return r;
    }

    private double findNearestEllipse(double x, double y) {
        int dNearest = -1;
        double d = 2*EllipticalR(new Point(x, y), avgEllipse);

        for (int dRes: dResults)
            if (Math.abs(dRes-d) < touchReactDiffPixel && Math.abs(dRes-d) < Math.abs(dNearest-d))
                dNearest = dRes;

        return dNearest;
    }

    private class ProcessImageTask extends AsyncTask<Uri, String, Boolean> {
        protected void onPreExecute() {
            buttonReset.setEnabled(false);
            toggleButtonMode.setEnabled(false);
        }

        protected Boolean doInBackground(Uri... params) {
            publishProgress(getResources().getString(R.string.process_image_task__loading));

            Uri uri = params[0];
            InputStream inStream;

            try {
                inStream = getContentResolver().openInputStream(uri);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                return false;
            }

            Mat sobel = new Mat(), blur = new Mat();

            try {
                imColor = readInputStreamIntoMat(inStream);
            } catch (IOException e) {
                return false;
            }

            publishProgress(getResources().getString(R.string.process_image_task__preprocess));
            CppPreprocess(imColor.getNativeObjAddr(), (imGray = new Mat()).getNativeObjAddr());
            Imgproc.GaussianBlur(imGray, blur, new Size(7, 7), 3);
            Point pt = QuickFindCenter(blur);

            publishProgress(getResources().getString(R.string.process_image_task__sobel));
            CppGetSobel(blur.getNativeObjAddr(), sobel.getNativeObjAddr());

            publishProgress(getResources().getString(R.string.process_image_task__erode_dilate));
            DynamicErode(sobel, pt);
            DynamicDilate(sobel, pt);

            publishProgress(getResources().getString(R.string.process_image_task__find_ellipse));
            avgEllipse = GetRBox(sobel);

            publishProgress(getResources().getString(R.string.process_image_task__integrate));
            double[] seq = EllipticalIntegrate(blur, avgEllipse);
            double[] de2 = secondDerivative(seq);

            double sum = 0.0;
            for (double i : seq) sum += i;
            double val_threshold = 0.8 * sum / seq.length;

            dResults.clear();
            for (int i=0, j=0; j<de2.length; i=j) {
                while (j<de2.length && de2[j]<-derivative2Threshold) j++;

                if (j > i && (seq[i] > val_threshold || seq[j] > val_threshold)) {
                    int d = i + j;
                    if (!dResults.isEmpty()) {
                        int last_d = dResults.get(dResults.size() - 1);
                        if (d - last_d < gapWidthThreshold * 2)
                            dResults.set(dResults.size() - 1, (d + last_d) / 2);
                        else
                            dResults.add(d);
                    } else
                        dResults.add(d);
                }

                while (j<de2.length && de2[j]>=-derivative2Threshold) j++;
            }

            return true;
        }

        protected void onProgressUpdate(String... progress) {
            textViewMessage.setText(progress[0]);
        }

        protected void onPostExecute(Boolean result) {
            buttonReset.setEnabled(result);
            toggleButtonMode.setEnabled(result);

            if (result) {
                Bitmap bitmap = Bitmap.createBitmap(imColor.cols(), imColor.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(imColor, bitmap);
                imageViewDisplay.setImageBitmap(bitmap);
                textViewMessage.setText(R.string.process_image_task__done);
                resetImage();
            } else
                textViewMessage.setText(R.string.process_image_task__failed);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        buttonLoad = (Button)findViewById(R.id.buttonLoad);
        buttonReset = (Button)findViewById(R.id.buttonReset);
        toggleButtonMode = (ToggleButton)findViewById(R.id.toggleButtonMode);
        textViewMessage = (TextView)findViewById(R.id.textViewMessage);
        textViewInfo = (TextView)findViewById(R.id.textViewInfo);
        imageViewDisplay = (ImageView)findViewById(R.id.imageViewDisplay);
        mAttacher = new PhotoViewAttacher(imageViewDisplay);

        buttonLoad.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 0);
            }
        });

        buttonReset.setEnabled(false);
        buttonReset.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                resetImage();
            }
        });

        final PhotoViewAttacher.OnPhotoTapListener autoModeTapListener = new PhotoViewAttacher.OnPhotoTapListener() {
            @Override
            public void onPhotoTap(View arg0, float arg1, float arg2) {
                double x = arg1 * imGray.cols(), y = arg2 * imGray.rows();
                double d = findNearestEllipse(x, y);

                double ratio = avgEllipse.size.width / avgEllipse.size.height;
                Mat dis = imColor.clone();
                Imgproc.circle(dis, avgEllipse.center, 2, new Scalar(255, 0, 0), -1);

                if (d > 0) {
                    RotatedRect box = new RotatedRect(avgEllipse.center, new Size(d, d/ratio), avgEllipse.angle);
                    Imgproc.ellipse(dis, box, new Scalar(255, 0, 0), 2);
                    String s = getResources().getString(R.string.textview_info_output,
                            box.center.x, box.center.y, box.size.width / 2, box.size.height / 2);
                    textViewInfo.setText(s);
                }

                Bitmap bitmap = Bitmap.createBitmap(dis.cols(), dis.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(dis, bitmap);
                imageViewDisplay.setImageBitmap(bitmap);
            }
        };

        final PhotoViewAttacher.OnPhotoTapListener customModeTapListener = new PhotoViewAttacher.OnPhotoTapListener() {
            @Override
            public void onPhotoTap(View arg0, float arg1, float arg2) {
                double x = arg1 * imGray.cols(), y = arg2 * imGray.rows();
                boolean found = false;

                for (Iterator<Point> it = selectPoints.iterator(); it.hasNext();) {
                    Point p = it.next();
                    if (Math.hypot(p.x-x, p.y-y) <= touchReactDiffPixel) {
                        it.remove();
                        found = true;
                        break;
                    }
                }

                if (!found) selectPoints.add(new Point(x, y));

                Mat dis = imColor.clone();
                for (Point pt: selectPoints) {
                    Imgproc.circle(dis, pt, 4, new Scalar(255, 255, 255), -1);
                }

                if (selectPoints.size() >= 5) {
                    MatOfPoint2f mop = new MatOfPoint2f();
                    mop.fromList(selectPoints);
                    RotatedRect box = Imgproc.fitEllipse(mop);
                    Imgproc.ellipse(dis, box, new Scalar(255, 0, 0), 1);
                    Imgproc.circle(dis, box.center, 2, new Scalar(255, 0, 0), -1);
                    String s = getResources().getString(R.string.textview_info_output,
                            box.center.x, box.center.y, box.size.width / 2, box.size.height / 2);
                    textViewInfo.setText(s);
                }

                Bitmap bitmap = Bitmap.createBitmap(dis.cols(), dis.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(dis, bitmap);
                imageViewDisplay.setImageBitmap(bitmap);
            }
        };

        mAttacher.setMinimumScale(0.1F);
        mAttacher.setMaximumScale(8.0F);
        mAttacher.setOnPhotoTapListener(autoModeTapListener);

        toggleButtonMode.setEnabled(false);
        toggleButtonMode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                resetImage();
                mAttacher.setOnPhotoTapListener(isChecked ? customModeTapListener : autoModeTapListener);
            }
        });
    }

    @Override
    protected void onActivityResult(int reqCode, int resCode, Intent data) {
        if(resCode == Activity.RESULT_OK && data != null)
            new ProcessImageTask().execute(data.getData());
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("Hello", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("Hello", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    // from: http://stackoverflow.com/questions/29232220/android-read-image-using-opencv
    private static Mat readInputStreamIntoMat(InputStream inputStream) throws IOException {
        byte[] temporaryImageInMemory = readStream(inputStream);
        return Imgcodecs.imdecode(new MatOfByte(temporaryImageInMemory), Imgcodecs.IMREAD_COLOR);
    }

    private static byte[] readStream(InputStream stream) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[16384];

        while ((nRead = stream.read(data, 0, data.length)) != -1)
            buffer.write(data, 0, nRead);

        buffer.flush();
        byte[] temporaryImageInMemory = buffer.toByteArray();
        buffer.close();
        stream.close();
        return temporaryImageInMemory;
    }
}
