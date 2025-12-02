import React from 'react';
import { TouchableOpacity, Text } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { styles } from '../styles/styles';

const CustomButton = ({
  title,
  onPress,
  iconName,
  variant = 'outline', // 'outline' or 'primary'
}) => {
  const isPrimary = variant === 'primary';

  // Select styles based on variant
  const containerStyle = isPrimary ? styles.btnPrimary : styles.btnOutline;
  const textStyle = isPrimary ? styles.btnTextPrimary : styles.btnTextOutline;
  const iconColor = isPrimary ? '#0F172A' : '#334155';

  return (
    <TouchableOpacity
      style={[styles.btnBase, containerStyle]}
      onPress={onPress}
      activeOpacity={0.7}>
      {iconName && <MaterialIcons name={iconName} size={24} color={iconColor} />}
      <Text style={textStyle}>{title}</Text>
    </TouchableOpacity>
  );
};

export default CustomButton;
