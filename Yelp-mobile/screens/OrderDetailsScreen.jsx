import React, { useState, useContext } from 'react';
import {
  View,
  Text,
  TextInput,
  Image,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  StyleSheet,
  StatusBar,
  Platform,
  Alert,
} from 'react-native';
import * as Location from 'expo-location';
import { MaterialIcons } from '@expo/vector-icons';
import { styles, COLORS } from '../styles/styles';
import { useNavigation } from '@react-navigation/native';
import DateTimePicker from '@react-native-community/datetimepicker';

import { AppContext } from '../context/AppContext';
import LoadingScreen from '../components/LoadingScreen';

import { SearchImage } from '../services/Endpoints';

export default function ScheduleOrderScreen({ route }) {
  const navigation = useNavigation();
  const [location, setLocation] = useState({ latitude: '', longitude: '', city: '' });
  const [date, setDate] = useState(new Date());
  const [showDate, setShowDate] = useState(false);
  const [showTime, setShowTime] = useState(false);

  const { isLoading, setIsLoading, setBusiness } = useContext(AppContext);

  const onChangeCity = (text) => {
    setLocation({
      ...location,
      city: text,
    });
  };

  // Updated onChange to handle Platform logic
  const onChange = (event, selectedDate) => {
    const currentDate = selectedDate || date;
    setDate(currentDate);

    // Only close the modals automatically on Android
    if (Platform.OS === 'android') {
      setShowDate(false);
      setShowTime(false);
    }
  };

  async function getCurrentLocation() {
    setIsLoading(true);
    let { status } = await Location.requestForegroundPermissionsAsync();
    if (status !== 'granted') {
      return;
    }

    try {
      let currentLocation = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
        timeout: 5000,
      });
      let addressResponse = await Location.reverseGeocodeAsync({
        latitude: currentLocation.coords.latitude,
        longitude: currentLocation.coords.longitude,
      });

      setLocation((current) => ({
        ...current,
        latitude: currentLocation.coords.latitude.toString(),
        longitude: currentLocation.coords.longitude.toString(),
        city: addressResponse[0].city,
      }));
    } catch {
      let lastKnown = await Location.getLastKnownPositionAsync();
      if (lastKnown) {
        setLocation((current) => ({
          ...current,
          latitude: lastKnown.coords.latitude.toString(),
          longitude: lastKnown.coords.longitude.toString(),
        }));
      } else {
      }
    } finally {
      setIsLoading(false);
    }
  }
  const { description, image } = route.params;

  async function onConfirmAndProceed() {
    // Validation
    if (location.city.length === 0) {
      Alert.alert('Missing Information', 'Please provide your current location or enter a city');
      return;
    }

    setIsLoading(true);

    try {
      // 1. Call the service
      // If this fails (e.g. 400, 500), it will jump straight to 'catch'
      let res = await SearchImage(
        image,
        description,
        location?.city || '',
        location?.latitude || '',
        location?.longitude || '',
        date?.toString() || '',
        date?.toString() || ''
      );

      // Optional: Double check if businesses array exists
      if (!res.businesses || res.businesses.length === 0) {
        Alert.alert('No Results', 'We could not find any restaurants matching that description.');
        return;
      }

      setBusiness(res.businesses);
      navigation.navigate('ChooseRestaurant');
    } catch (err) {
      console.error('Search failed:', err);

      const errorMessage = err.message || 'Something went wrong';

      Alert.alert('Request Failed', errorMessage);
    } finally {
      setIsLoading(false);
    }
  }
  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.backgroundLight} />

      {isLoading ? <LoadingScreen /> : null}

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.sectionContainer}>
          <Text style={[styles.restaurantName, { fontSize: 16, marginBottom: 12 }]}>Your Item</Text>

          <View style={styles.itemCard}>
            {image ? (
              <Image source={{ uri: image }} style={styles.itemImage} />
            ) : (
              <View style={styles.itemTextContainer}>
                <MaterialIcons
                  name="restaurant"
                  size={32}
                  color={COLORS.primary}
                  style={{ marginBottom: 8 }}
                />
                <Text style={styles.itemDescriptionText}>{description}</Text>
              </View>
            )}

            {/* Optional: Add a subtle text overlay or badge if needed, otherwise clean look */}
          </View>
        </View>

        <View style={{ marginBottom: 24 }}>
          <Text style={[styles.restaurantName, { fontSize: 16, marginBottom: 12 }]}>Location</Text>

          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.textInput}
              placeholder="Enter your city"
              placeholderTextColor={COLORS.textSecondary}
              value={location?.city}
              onChangeText={(text) => onChangeCity(text)}
            />
            <MaterialIcons name="search" size={24} color={COLORS.textSecondary} />
          </View>

          <TouchableOpacity style={styles.locationButton} onPress={() => getCurrentLocation()}>
            <MaterialIcons name="my-location" size={20} color={COLORS.primary} />
            <Text style={[styles.btnTextOutline, { color: COLORS.primary, fontSize: 14 }]}>
              Use Current Location
            </Text>
          </TouchableOpacity>
        </View>

        <View style={{ marginBottom: 24 }}>
          <Text style={[styles.restaurantName, { fontSize: 16, marginBottom: 12 }]}>
            Date & Time
          </Text>

          {Platform.OS === 'ios' && (
            <View style={styles.iosPickerContainer}>
              <DateTimePicker
                testID="dateTimePickerDate"
                value={date}
                mode={'date'}
                display="compact"
                onChange={onChange}
                style={{ marginRight: 10 }}
              />
              <DateTimePicker
                testID="dateTimePickerTime"
                value={date}
                mode={'time'}
                display="compact"
                onChange={onChange}
              />
            </View>
          )}

          {Platform.OS === 'android' && (
            <View style={styles.androidRow}>
              <TouchableOpacity
                onPress={() => setShowDate(true)}
                style={[styles.inputWrapper, { flex: 1, marginRight: 8 }]}>
                <Text style={styles.dateText}>{date.toLocaleDateString()}</Text>
                <MaterialIcons name="date-range" size={20} color={COLORS.textSecondary} />
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => setShowTime(true)}
                style={[styles.inputWrapper, { flex: 1, marginLeft: 8 }]}>
                <Text style={styles.dateText}>
                  {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </Text>
                <MaterialIcons name="access-time" size={20} color={COLORS.textSecondary} />
              </TouchableOpacity>

              {showDate && (
                <DateTimePicker
                  testID="dateTimePickerDate"
                  value={date}
                  mode={'date'}
                  is24Hour={true}
                  onChange={onChange}
                />
              )}
              {showTime && (
                <DateTimePicker
                  testID="dateTimePickerTime"
                  value={date}
                  mode={'time'}
                  is24Hour={true}
                  onChange={onChange}
                />
              )}
            </View>
          )}

          <Text style={[styles.metaText, { textAlign: 'center', marginTop: 12 }]}>
            We will show shops available during this time
          </Text>
        </View>
      </ScrollView>

      <View style={styles.bottomFixedFooter}>
        <TouchableOpacity
          style={[styles.btnBase, styles.btnPrimary]}
          onPress={() => onConfirmAndProceed(description, '', '', date, date)}>
          <Text style={styles.btnTextPrimary}>Confirm & Proceed</Text>
          <MaterialIcons name="arrow-forward" size={24} color="#FFF" />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}
