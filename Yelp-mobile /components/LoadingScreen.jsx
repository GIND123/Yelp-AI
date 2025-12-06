import { ActivityIndicator, View, Modal } from 'react-native';

import { COLORS } from '../styles/styles';

// Reusable loading component
function LoadingScreen({ visible = true }) {
  return (
    <Modal
      transparent={true}
      animationType="fade"
      visible={visible}
      onRequestClose={() => {
        // Prevent closing on Android back button
      }}>
      <View
        style={{
          flex: 1,
          backgroundColor: 'rgba(0, 0, 0, 0.5)', // Semi-transparent overlay
          justifyContent: 'center',
          alignItems: 'center',
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
    </Modal>
  );
}

export default LoadingScreen;
