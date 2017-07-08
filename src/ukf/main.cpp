#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <uWS/uWS.h>

#include "json/json.hpp"

#include "ukf/tools.h"
#include "ukf/ukf.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// for convenience
using json = nlohmann::json;

struct State {
  ukf::Ukf ukf;
  vector<Eigen::Array4d> estimations;
  vector<Eigen::Array4d> ground_truth;
};

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(const string& s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  h.onMessage([](uWS::WebSocket<uWS::SERVER> ws, char* data, size_t length,
                 uWS::OpCode opCode) {
    State* state = static_cast<State*>(ws.getUserData());

    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (!length || length <= 2 || data[0] != '4' || data[1] != '2') return;

    auto s = hasData(string(data));
    if (s.empty()) {
      string msg = "42[\"manual\",{}]";
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      return;
    }

    auto j = json::parse(s);

    string event = j[0].get<string>();

    if (event == "telemetry") {
      string sensor_measurment = j[1]["sensor_measurement"];
      std::istringstream iss(sensor_measurment);

      string sensor_type;
      iss >> sensor_type;

      if (sensor_type == "L") {
        LidarMeasurement m;
        iss >> m.data(0);
        iss >> m.data(1);
        iss >> m.timestamp;

        state->ukf.ProcessMeasurement(m);
      } else if (sensor_type == "R") {
        RadarMeasurement m;
        iss >> m.data(0);
        iss >> m.data(1);
        iss >> m.data(2);
        iss >> m.timestamp;

        state->ukf.ProcessMeasurement(m);
      } else {
        cerr << "Wrong sensor type" << endl;
      }

      auto estimate = state->ukf.Estimate();
      state->estimations.emplace_back(estimate);

      Eigen::Array4d gt;
      iss >> gt(0) >> gt(1) >> gt(2) >> gt(3);
      state->ground_truth.emplace_back(std::move(gt));

      auto rmse = CalculateRmse(state->estimations, state->ground_truth);

      json msgJson;
      msgJson["estimate_x"] = estimate(0);
      msgJson["estimate_y"] = estimate(1);
      msgJson["rmse_x"] = rmse(0);
      msgJson["rmse_y"] = rmse(1);
      msgJson["rmse_vx"] = rmse(2);
      msgJson["rmse_vy"] = rmse(3);
      auto msg = "42[\"estimate_marker\"," + msgJson.dump() + "]";
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse* res, uWS::HttpRequest req, char* data,
                     size_t, size_t) {
    const string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    ws.setUserData(new State);
    cout << "Connected!!!" << endl;
  });

  h.onDisconnection([](uWS::WebSocket<uWS::SERVER> ws, int code, char* message,
                       size_t length) {
    delete static_cast<State*>(ws.getUserData());
    cout << "Disconnected" << endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    cout << "Listening to port " << port << endl;
  } else {
    cerr << "Failed to listen to port" << endl;
    return -1;
  }
  h.run();
}
