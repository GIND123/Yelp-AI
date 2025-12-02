import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  SafeAreaView,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
  Alert,
  Image,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { styles } from '../styles/styles';
import * as ImagePicker from 'expo-image-picker';
import { useNavigation } from '@react-navigation/native';
import CustomButton from '../components/CustomButton';

export default function AddPhoto() {
  const [description, setDescription] = useState('');
  const [image, setImage] = useState(null);
  const navigation = useNavigation();

  const clickImage = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert('Permission required', 'Permission to access the camera is required.');
      return;
    }

    // 2. Use launchCameraAsync instead of launchImageLibraryAsync
    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permissionResult.granted) {
      Alert.alert('Permission required', 'Permission to access the media library is required.');
      return;
    }

    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  let onPressNext = () => {
    if (description.length === 0 && image === null) {
      Alert.alert(
        'Missing Information',
        'Please provide a description or upload an image to continue.'
      );
      return;
    }
    navigation.navigate('ScheduleOrderScreen', {
      description: description,
      image: image,
    });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      {/* <CameraView style={styles.camera} facing={facing} /> */}
      <StatusBar barStyle="dark-content" backgroundColor="#F6F8F6" />
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={{ flex: 1 }}>
        <View style={styles.container}>
          <View showsVerticalScrollIndicator={false}>
            {/* Dashed Placeholder */}
            {!image ? (
              <View style={styles.uploadPlaceholder}>
                <MaterialIcons name="image" size={64} color="#94A3B8" />
                <View style={{ alignItems: 'center', gap: 4 }}>
                  <Text style={styles.uploadPlaceholderTextTitle}>Add a photo of your food</Text>
                  <Text style={styles.uploadPlaceholderTextSub}>
                    Take a new picture or select one from your gallery.
                  </Text>
                </View>
              </View>
            ) : (
              <Image src={image} style={styles.uploadPlaceholder} />
            )}

            {/* Action Buttons Grid */}
            <View style={styles.buttonGrid}>
              <CustomButton
                title="Take Photo"
                iconName="photo-camera"
                variant="outline"
                onPress={() => clickImage()}
              />
              <CustomButton
                title="From Gallery"
                iconName="photo-library"
                variant="outline"
                onPress={() => pickImage()}
              />
            </View>

            {/* Description Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Description</Text>
              <TextInput
                style={styles.textArea}
                placeholder="e.g. Delicious salad from my favorite cafe..."
                placeholderTextColor="#94A3B8"
                multiline={true}
                numberOfLines={4}
                value={description}
                onChangeText={setDescription}
              />
            </View>
          </View>

          {/* Footer */}
          <View style={styles.footer}>
            <CustomButton title="Next" variant="primary" onPress={() => onPressNext()} />
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
