import { Alert } from 'react-native';

export const throwErrorAlert = (title, message) => {
  Alert.alert(title, message, [{ text: 'OK', style: 'cancel' }], { cancelable: true });
};
