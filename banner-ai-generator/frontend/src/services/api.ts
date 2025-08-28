import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'

// Thêm xử lý hợp lý cho rate limiting
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 giây

// Sử dụng proxy thông qua Vite
const api: AxiosInstance = axios.create({
  baseURL: '/api', // Không thay đổi, sử dụng proxy của Vite
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Thêm xử lý rate limiting
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error.config;
    
    // Không có cấu hình hoặc đã retry quá số lần -> reject
    if (!config || config.__retryCount >= MAX_RETRIES) {
      return Promise.reject(error);
    }

    // Lỗi 429 -> retry
    if (error.response?.status === 429) {
      // Tăng số lần retry
      config.__retryCount = config.__retryCount || 0;
      config.__retryCount += 1;

      // Tạo promise delay
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
      
      // Thử lại request
      return api(config);
    }

    return Promise.reject(error);
  }
);

// Request interceptor for auth and logging
api.interceptors.request.use(
  (config) => {
    // Add timestamp to prevent caching
    if (config.method === 'get') {
      config.params = {
        ...config.params,
        _t: Date.now()
      }
    }

    // Log requests in development
    if (import.meta.env.DEV) {
      console.log(`🚀 ${config.method?.toUpperCase()} ${config.url}`, config.data || config.params)
    }

    return config
  },
  (error) => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// [phần code khác giữ nguyên]
// ... code tiếp theo giữ nguyên

export const apiService = {
  // [giữ nguyên các phương thức]
}

export { api }
export default apiService
