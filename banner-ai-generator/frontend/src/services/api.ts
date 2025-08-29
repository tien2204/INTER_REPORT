import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'

// Create axios instance with default config
const api: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

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

// Response interceptor for error handling
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log responses in development
    if (import.meta.env.DEV) {
      console.log(`✅ ${response.config.method?.toUpperCase()} ${response.config.url}`, response.data)
    }
    
    return response
  },
  (error) => {
    // Log errors
    if (error.response) {
      console.error(`❌ ${error.response.status} ${error.response.config?.method?.toUpperCase()} ${error.response.config?.url}`, error.response.data)
    } else if (error.request) {
      console.error('Network error:', error.request)
    } else {
      console.error('Request setup error:', error.message)
    }

    // Handle common HTTP errors
    if (error.response?.status === 401) {
      // Handle unauthorized
      console.warn('Unauthorized request - consider implementing auth')
    } else if (error.response?.status === 403) {
      // Handle forbidden
      console.warn('Forbidden request')
    } else if (error.response?.status === 404) {
      // Handle not found
      console.warn('Resource not found')
    } else if (error.response?.status >= 500) {
      // Handle server errors
      console.error('Server error occurred')
    }

    return Promise.reject(error)
  }
)

// API service methods
export const apiService = {
  // GET request
  get: <T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> => {
    return api.get(url, config)
  },

  // POST request
  post: <T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> => {
    return api.post(url, data, config)
  },

  // PUT request
  put: <T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> => {
    return api.put(url, data, config)
  },

  // PATCH request
  patch: <T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> => {
    return api.patch(url, data, config)
  },

  // DELETE request
  delete: <T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> => {
    return api.delete(url, config)
  },

  // Upload file with progress
  upload: <T = any>(
    url: string, 
    formData: FormData, 
    onProgress?: (progressEvent: any) => void
  ): Promise<AxiosResponse<T>> => {
    return api.post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: onProgress,
    })
  },

  // Download file
  download: (url: string, filename?: string): Promise<void> => {
    return api.get(url, {
      responseType: 'blob'
    }).then(response => {
      // Create blob link to download
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      
      // Try to get filename from content-disposition header
      const contentDisposition = response.headers['content-disposition']
      let downloadFilename = filename
      
      if (contentDisposition) {
        const matches = /filename="([^"]*)"/.exec(contentDisposition)
        if (matches != null && matches[1]) {
          downloadFilename = matches[1]
        }
      }
      
      if (downloadFilename) {
        link.setAttribute('download', downloadFilename)
      }
      
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    })
  }
}

// Export axios instance for direct use if needed
export { api }
export default apiService
