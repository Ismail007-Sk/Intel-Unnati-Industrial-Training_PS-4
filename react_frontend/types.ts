
export type Role = 'admin' | 'doctor' | 'nurse' | 'patient';

export interface UserProfile {
  username: string;
  role: Role;
}

export interface UploadedFile {
  id: string;
  name: string;
  owner: string;
  visibility: Role;
  timestamp: string;
}
