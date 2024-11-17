package com.example.caloriepredictorapp;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONException;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {

    private EditText inputProtein, inputFat, inputCarbs;
    private TextView resultTextView;
    private Button calculateButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inputProtein = findViewById(R.id.inputProtein);
        inputFat = findViewById(R.id.inputFat);
        inputCarbs = findViewById(R.id.inputCarbs);
        resultTextView = findViewById(R.id.resultTextView);
        calculateButton = findViewById(R.id.calculateButton);

        calculateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                calculateCalories();
            }
        });
    }

    private void calculateCalories() {
        String protein = inputProtein.getText().toString();
        String fat = inputFat.getText().toString();
        String carbs = inputCarbs.getText().toString();

        if (protein.isEmpty() || fat.isEmpty() || carbs.isEmpty()) {
            resultTextView.setText("Please fill all fields.");
            return;
        }

        String url = "http://192.168.66.193:5000/predict";

        JSONObject requestData = new JSONObject();
        try {
            requestData.put("Protein", Double.parseDouble(protein));
            requestData.put("Fat", Double.parseDouble(fat));
            requestData.put("Carbs", Double.parseDouble(carbs));
        } catch (JSONException e) {
            e.printStackTrace();
            return;
        }

        RequestQueue queue = Volley.newRequestQueue(this);
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
                Request.Method.POST, url, requestData,
                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {
                        try {
                            double predictedCalories = response.getDouble("Predicted Calories");
                            resultTextView.setText("Predicted Calories: " + predictedCalories);
                        } catch (JSONException e) {
                            e.printStackTrace();
                            resultTextView.setText("Error parsing response.");
                        }
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        error.printStackTrace();
                        resultTextView.setText("Error connecting to backend.");
                    }
                });

        queue.add(jsonObjectRequest);
    }
}
