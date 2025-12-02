import { ActivityIndicator, View } from 'react-native';

import { COLORS } from '../styles/styles';

// Reusable loading component
function LoadingScreen() {
  return (
    <View
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)', // Semi-transparent overlay
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 9999, // Ensures it's on top
      }}>
      <View
        style={{
          padding: 20,
          borderRadius: 10,
          backgroundColor: 'rgba(255, 255, 255, 0.9)', // Optional: white background for spinner
        }}>
        <ActivityIndicator size="large" color={COLORS.primary} />
      </View>
    </View>
  );
}

export default LoadingScreen;
