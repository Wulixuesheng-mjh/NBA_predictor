/** Axios-based API client with common configuration. */

import axios from "axios";

const apiClient = axios.create({
  baseURL: "/api/v1",
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor: inject admin API key from localStorage
apiClient.interceptors.request.use((config) => {
  const key = localStorage.getItem("admin_api_key");
  if (key && config.headers) {
    config.headers["X-API-Key"] = key;
  }
  return config;
});

// Response interceptor: unwrap common response wrapping
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Surface 401 auth errors clearly for admin endpoints
    if (error.response?.status === 401) {
      console.error("Admin auth failed — check API key in Admin page");
    }
    const message =
      error.response?.data?.error?.message ||
      error.message ||
      "Unknown error occurred";
    console.error("API Error:", message);
    return Promise.reject(error);
  }
);

export default apiClient;
