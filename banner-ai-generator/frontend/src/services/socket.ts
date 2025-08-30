// frontend/src/services/socket.ts
import { io, Socket } from 'socket.io-client';

let socket: Socket | null = null;

// Cấu hình kết nối tối ưu
const SOCKET_CONFIG = {
  path: '/socket.io/',
  transports: ['websocket', 'polling'], // Thêm polling làm fallback
  upgrade: true,
  rememberUpgrade: true,
  timeout: 20000,
  reconnection: true,
  reconnectionAttempts: 5,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  maxReconnectionAttempts: 5,
  forceNew: false
};

export const connectSocket = (): Socket => {
  if (socket?.connected) {
    console.log('✅ Socket already connected:', socket.id);
    return socket;
  }

  console.log('🔄 Connecting to Socket.IO...');
  
  // Tạo kết nối socket mới
  socket = io(SOCKET_CONFIG);

  // Connection events
  socket.on('connect', () => {
    console.log('✅ Socket connected:', socket?.id);
  });

  socket.on('disconnect', (reason) => {
    console.log('❌ Socket disconnected:', reason);
  });

  socket.on('connect_error', (error) => {
    console.error('❌ Socket connection error:', error.message);
    
    // Log chi tiết hơn cho debugging
    if (error.message.includes('websocket')) {
      console.log('🔄 WebSocket failed, falling back to polling...');
    }
  });

  socket.on('reconnect', (attemptNumber) => {
    console.log('✅ Socket reconnected after', attemptNumber, 'attempts');
  });

  socket.on('reconnect_attempt', (attemptNumber) => {
    console.log('🔄 Socket reconnection attempt:', attemptNumber);
  });

  socket.on('reconnect_error', (error) => {
    console.error('❌ Socket reconnection error:', error.message);
  });

  socket.on('reconnect_failed', () => {
    console.error('💀 Socket reconnection failed after all attempts');
  });

  // Custom events
  socket.on('welcome', (data) => {
    console.log('👋 Welcome message:', data.message);
  });

  socket.on('error', (data) => {
    console.error('❌ Server error:', data.message);
  });

  return socket;
};

export const getSocket = (): Socket | null => socket;

export const disconnectSocket = (): void => {
  if (socket) {
    console.log('🔌 Disconnecting socket...');
    socket.disconnect();
    socket = null;
  }
};

// Helper function để emit events với error handling
export const emitEvent = (event: string, data: any): void => {
  if (socket?.connected) {
    socket.emit(event, data);
    console.log('📤 Emitted event:', event, data);
  } else {
    console.warn('⚠️ Cannot emit event - socket not connected:', event);
  }
};

// Helper function để subscribe to progress
export const subscribeToProgress = (designId: string): void => {
  if (socket?.connected) {
    socket.emit('subscribe_to_progress', designId);
    console.log('📡 Subscribed to progress for design:', designId);
  } else {
    console.warn('⚠️ Cannot subscribe - socket not connected');
  }
};
